from json import tool
# Self-Reasoning Agents with Tools
import os

from dotenv import load_dotenv
import langchain
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic.types import SecretStr


# Load environment variables from .env file
load_dotenv()


# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVELY_SEARCH_API_KEY")


if not OPENAI_API_KEY:
    raise ValueError("API key not found. Make sure .env file is set correctly.")


# Initialize ChatOpenAI
llm_obj = ChatOpenAI(
    api_key=SecretStr(OPENAI_API_KEY),
    model="gpt-3.5-turbo",
    temperature=0.7,
)

def get_document_from_web(url: str) -> list[Document]:
    loader = WebBaseLoader(web_path=url)
    docs = loader.load()
    spliter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    split_docs = spliter.split_documents(docs)
    return split_docs


def create_db(docs: list[Document]):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

def create_agent():
    prompt_obj = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant called Max."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    docs = get_document_from_web(
        url="https://byte93.pythonanywhere.com/articles/articles/biblioteka-asyncio"
    )
    vectorStore = create_db(docs=docs)
    retriever = vectorStore.as_retriever()

    search_tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="lcel_search",
        description="Use this tool when searching information about Asyncio"
    )

    tools = [search_tool, retriever_tool]

    agent_obj = create_openai_functions_agent(
        llm=llm_obj,
        prompt=prompt_obj,
        tools=tools
    )

    agentExecutor = AgentExecutor(
        agent=agent_obj,
        tools=tools
    )
    return agentExecutor



if __name__ == "__main__":
    agent = create_agent()
    chat_history = [
    ]
    while True:
        user_input = str(input("Input: \n"))    # Що таке Asyncio?
        if user_input.lower() == "exit":
            break

        response = agent.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.get('output', "")))
        print(f"Output: \n {response.get('output')}")
