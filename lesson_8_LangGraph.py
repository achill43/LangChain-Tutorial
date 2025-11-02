from langchain_openai import ChatOpenAI  # OpenAI LLM interface
from crewai_tools import SerperDevTool  # Web search tool
from langchain.tools import tool  # Decorator to turn functions into tools
from langgraph.prebuilt import create_react_agent  # Helper to build ReAct-style agents
from langgraph.graph import START, StateGraph, END  # Core components of LangGraph
from typing import TypedDict  # Used to define structured state
import os
from dotenv import load_dotenv
from pydantic.types import SecretStr

load_dotenv()  # Load environment variables from .env file


# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVELY_SEARCH_API_KEY")


if not OPENAI_API_KEY:
    raise ValueError("API key not found. Make sure .env file is set correctly.")


# Initialize ChatOpenAI
llm = ChatOpenAI(
    api_key=SecretStr(OPENAI_API_KEY),
    model="gpt-3.5-turbo",
    temperature=0.7,
)


# --- Tool Definition ---
@tool
def serper_search(user_query: str) -> str:
    """
    Perform a real-time search using the Serper API.

    This tool takes a plain-text user query, sends it to Serper (a web search API),
    and returns a string with the top relevant results. It can be used by agents
    to gather up-to-date information from the internet as part of a reasoning or
    research task.

    Args:
        user_query (str): A natural language search prompt.

    Returns:
        str: A formatted string of search results from Serper.
    """
    return SerperDevTool().run(query=user_query)


# --- Define State ---
class AgentState(TypedDict):
    user_query: str
    answer: str


# --- Define Node ---
def search_agent(state: AgentState) -> str:
    """
    Executes a ReAct-style agent that processes a user query.

    This function takes the current state (which includes the user's question),
    creates an agent using the Gemini language model and the `serper_search` tool,
    then runs the agent to get a response. The final answer is returned as updated state.

    Args:
        state (AgentState): A dictionary with the user's query.

    Returns:
        dict: Updated state with the generated answer.
    """
    agent = create_react_agent(llm, [serper_search])
    result = agent.invoke({"messages": state["user_query"]})
    return {"answer": result["messages"][-1].content}


# --- Define Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("search_agent", search_agent)

workflow.add_edge(START, "search_agent")
workflow.add_edge("search_agent", END)

app = workflow.compile()


from IPython.display import Image, display

Image(app.get_graph().draw_mermaid_png())


# --- Run Graph ---
result = app.invoke({"user_query": "What are the advantages of using Python as a tool in the current scenario?"})
print("User question:")
print("What are the advantages of using Python as a tool in the current scenario?")
print("AI assintent answer:")
print(result["answer"])
