import asyncio
import json
import os
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
github_token = os.getenv("GITHUB_TOKEN")


if not api_key:
    raise ValueError("API key not found. Make sure .env file is set correctly.")

if not github_token:
    raise ValueError("GITHUB_TOKEN not found. Make sure .env file is set correctly.")


# The pull request under review
OWNER = "achill43"
REPO = "STO_Project"
PULL_NUMBER = 2

# GitHub times out serving a review that carries too many inline comments.
MAX_INLINE_COMMENTS = 15

# GitHub's wording when APPROVE/REQUEST_CHANGES is used on your own pull request.
SELF_REVIEW_ERROR = "on your own pull request"

# Matches a diff hunk header, capturing the starting line of the new file version:
# @@ -12,7 +34,9 @@  ->  34
HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def unwrap(tool_result):
    """MCP tools return a list of content blocks; pull the JSON payload out."""
    if isinstance(tool_result, list):
        tool_result = tool_result[0]["text"]
    return json.loads(tool_result)


def added_lines(patch):
    """Line numbers in the new file version that GitHub accepts a comment on.

    GitHub rejects a review comment unless its line sits inside a diff hunk,
    which is what caused "Line could not be resolved". We walk the patch and
    keep only the lines the PR actually adds.
    """
    lines = []
    new_line = 0

    for row in patch.splitlines():
        header = HUNK_HEADER.match(row)
        if header:
            new_line = int(header.group(1))
        elif row.startswith("+"):
            lines.append(new_line)
            new_line += 1
        elif not row.startswith(("-", "\\")):
            # A context line: present in both versions, so it advances the counter.
            new_line += 1

    return lines


async def commentable_lines(tools):
    """Map every changed file to the line numbers a comment may be anchored to."""
    result = await tools["get_pull_request_files"].ainvoke(
        {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}
    )

    allowed = {}
    for changed_file in unwrap(result):
        # Binary files and very large diffs come back without a patch.
        patch = changed_file.get("patch")
        if patch:
            allowed[changed_file["filename"]] = added_lines(patch)

    return allowed


def guard_review_tool(tool, allowed, review_list_tool):
    """Make create_pull_request_review survive GitHub's failure modes.

    422 "Line could not be resolved": one comment on a line outside the diff
    fails the whole call, losing every other comment. We drop those first.

    Timeout: GitHub gives up serving a review that carries a large comments
    array, so we cap the batch. A timeout does not mean the review was not
    created, so before retrying we check whether it already landed -- blindly
    resubmitting is how you end up posting the same review twice.

    422 "Can not request changes on your own pull request": GitHub only allows
    APPROVE and REQUEST_CHANGES from someone other than the author. Since the
    token here belongs to the author, we fall back to COMMENT. A validation
    error means nothing was created, so this retry cannot double-post.
    """

    async def review_count():
        result = await review_list_tool.ainvoke(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}
        )
        return len(unwrap(result))

    async def submit_review(**kwargs):
        kept, dropped = [], []

        for comment in kwargs.get("comments") or []:
            path, line = comment.get("path"), comment.get("line")
            if line in allowed.get(path, []):
                kept.append(comment)
            else:
                dropped.append(f"{path}:{line}")

        notes = []
        if dropped:
            notes.append(
                f"Skipped {len(dropped)} comment(s) on lines outside the diff: "
                f"{', '.join(dropped)}"
            )

        if len(kept) > MAX_INLINE_COMMENTS:
            notes.append(
                f"Kept the first {MAX_INLINE_COMMENTS} of {len(kept)} comments; "
                "GitHub times out on larger reviews."
            )
            kept = kept[:MAX_INLINE_COMMENTS]

        kwargs["comments"] = kept

        before = await review_count()
        try:
            response = await tool.ainvoke(kwargs)
        except Exception as exc:
            # The review may have been created anyway -- check before retrying.
            if await review_count() > before:
                response = (
                    "GitHub timed out, but the review was created successfully. "
                    "Do not submit it again."
                )
                notes.append(f"Underlying error: {exc}")
            elif SELF_REVIEW_ERROR in str(exc):
                # Nothing was created, so downgrading and retrying is safe.
                notes.append(
                    f"GitHub rejected event={kwargs.get('event')!r} because you cannot "
                    "approve or request changes on your own pull request; "
                    "submitted as COMMENT instead."
                )
                kwargs["event"] = "COMMENT"
                response = await tool.ainvoke(kwargs)
            else:
                raise

        summary = f"Posted {len(kept)} inline comment(s)."
        return "\n\n".join([str(response), summary, *notes])

    return StructuredTool.from_function(
        coroutine=submit_review,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


async def main():
    model = ChatOpenAI(api_key=api_key, model="gpt-5o", temperature=0.2)

    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {
                    "PATH": os.environ["PATH"],
                    "GITHUB_PERSONAL_ACCESS_TOKEN": github_token,
                },
            }
        }
    )

    print("Loading GitHub MCP tools...")
    tools = {tool.name: tool for tool in await client.get_tools()}

    print("Reading the diff to find commentable lines...")
    allowed = await commentable_lines(tools)
    print(f"{len(allowed)} changed file(s) with a usable diff.\n")

    tools["create_pull_request_review"] = guard_review_tool(
        tools["create_pull_request_review"],
        allowed,
        tools["get_pull_request_reviews"],
    )

    # Enhanced System Prompt with explicit instructions for inline review comments
    system_prompt = SystemMessage(
        content=(
            "You are an expert Senior Code Reviewer. Your goal is to conduct line-by-line code reviews on GitHub Pull Requests.\n\n"
            "Workflow:\n"
            "1. Fetch the pull request details and full diff files using the available GitHub tools.\n"
            "2. Map each issue to a filename and a line number taken from the ALLOWED LINES list in the user message.\n"
            "3. Submit ONE create_pull_request_review call containing every inline comment in its 'comments' array, "
            "plus a top-level summary in 'body'.\n"
            "   Always use event='COMMENT'. The token belongs to the pull request author, and GitHub "
            "refuses APPROVE or REQUEST_CHANGES on your own pull request. State your verdict in the "
            "summary body instead.\n\n"
            "Line number rules:\n"
            "- GitHub only accepts a comment on a line the pull request actually changed.\n"
            "- Use ONLY the file paths and line numbers given in ALLOWED LINES. Never count lines yourself, "
            "and never take a line number from the full file contents.\n"
            "- If an issue is on a line that is not in the list, describe it in the summary body instead.\n\n"
            "Inline Commenting Guidelines:\n"
            "- Be concise, constructive, and direct.\n"
            "- Focus on actionable feedback: bugs, security risks, memory leaks, or missing edge cases.\n"
            "- Provide small, corrected code snippets in Markdown where applicable.\n"
            "- Do not comment on formatting-only changes such as added blank lines.\n"
            f"- Post at most {MAX_INLINE_COMMENTS} inline comments. Choose the most important "
            "issues and cover the rest in the summary body; GitHub rejects larger reviews.\n"
            "- If a tool reports that the review was already created, stop. Do not submit it again."
        )
    )

    agent = create_react_agent(model, list(tools.values()), prompt=system_prompt)

    # Prompt forcing granular inline analysis
    user_request = (
        f"Please review pull request #{PULL_NUMBER} in the repository '{OWNER}/{REPO}'. "
        "Fetch the diff, analyze the changed files, and leave inline code comments "
        "on specific lines where issues exist, then submit the review.\n\n"
        "ALLOWED LINES (the only file/line pairs GitHub will accept a comment on):\n"
        + json.dumps(allowed, indent=1)
    )

    print(f"Starting review process for: {user_request}\n")

    inputs = {"messages": [HumanMessage(content=user_request)]}
    async for chunk in agent.astream(inputs, stream_mode="values"):
        message = chunk["messages"][-1]
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
