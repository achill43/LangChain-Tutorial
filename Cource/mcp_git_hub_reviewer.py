import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
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


async def main():
    # 1. Initialize the LLM
    model = ChatOpenAI(api_key=api_key, model="gpt-4o", temperature=0.2)

    # 2. Connect to the GitHub MCP Server and load its tools
    # Using the standard official github MCP server package via npx
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
    # Wrap MCP server tools seamlessly into LangChain format
    tools = await client.get_tools()

    # 3. Define the System Instructions for the AI Reviewer
    system_prompt = SystemMessage(
        content=(
            "You are an expert Senior Code Reviewer. Your goal is to review the requested GitHub pull request.\n"
            "Workflow:\n"
            "1. Use the appropriate tool to fetch the pull request details or diff using the owner, repo, and pull_number.\n"
            "2. Analyze the code changes for bugs, architectural design, performance bottlenecks, and security issues.\n"
            "3. Use the review commenting tools to leave structured, polite, and actionable feedback directly on the PR."
        )
    )

    # 4. Create the LangChain Agent with the MCP Tools
    agent = create_react_agent(model, tools, prompt=system_prompt)

    # 5. Execute the review instruction
    # Change owner, repo, and pull_number to target your specific repository
    user_request = (
        "Please review pull request #5 in the repository 'achill43/STO_Project'. "
        "Fetch the diff, identify any improvements, and post a review summary comment on it."
    )

    print(f"Starting review for: {user_request}")

    # Run the agent stream
    inputs = {"messages": [HumanMessage(content=user_request)]}
    async for chunk in agent.astream(inputs, stream_mode="values"):
        message = chunk["messages"][-1]
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
