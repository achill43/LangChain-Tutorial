import os

from dotenv import load_dotenv

from langchain.agents import create_agent

from langchain_openai import ChatOpenAI

from langchain_community.utilities.github import GitHubAPIWrapper

from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit


# Load environment variables from .env file

load_dotenv()


GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")

GITHUB_APP_PRIVATE_KEY = os.getenv("GITHUB_APP_PRIVATE_KEY")

GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")


# Initialize the wrapper

github = GitHubAPIWrapper(
    github_app_id=GITHUB_APP_ID,
    github_app_private_key=GITHUB_APP_PRIVATE_KEY,
    github_repository=GITHUB_REPOSITORY,
)

# Create the toolkit

toolkit = GitHubToolkit.from_github_api_wrapper(github)

# Get the tools the agent can use

tools = toolkit.get_tools()


# Rename the tools to comply with OpenAI's function calling name pattern
for tool in tools:
    # Replace dots in the name with underscores to satisfy the pattern '^[a-zA-Z0-9_-]+$'
    tool.name = tool.name.replace(".", "_").replace("-", "_").replace(" ", "_").lower()

print("Reworking tools:")
for tool in tools:
    print(f"- {tool.name}")

# Initialize your LLM

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)


# Create the agent executor

system_prompt_str = "You are a senior GitHub code review assistant. Your output MUST follow the JSON schema strictly."

agent_executor = create_agent(llm, tools, system_prompt=system_prompt_str)


if __name__ == "__main__":

    # Ensure this PR ID exists and is open on achill43/LangChain-Tutorial

    review_task = "Analyze the code changes last pull request and post a constructive review comment. Focus on best practices."

    response = agent_executor.invoke({"input": review_task})

    print(response["output"])
