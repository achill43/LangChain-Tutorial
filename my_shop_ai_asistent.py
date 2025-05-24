# Long term SQLite Agent memory
import os
import asyncio

from dotenv import load_dotenv
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.memory import ConversationBufferMemory
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

from sqlalchemy import text
from docs_db_connection import AsyncSessionLocal

async def load_products_from_db() -> list[Document]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("SELECT model, name, description, price FROM products"))
        rows = result.fetchall()

        docs = []
        for row in rows:
            content = f"Модель: {row.model}\nНазва: {row.name}\nОпис: {row.description}\nЦіна: {row.price} грн"
            doc = Document(page_content=content, metadata={"source": "products_db"})
            docs.append(doc)
        return docs


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

def create_agent(session_id: str):
    # Create message history using SQLite
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///agent_memory.db"  # Replace with your desired database path
    )

    # Wrap it in buffer memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=history
    )

    prompt_obj = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly assistant called Max."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

     # Завантажуємо товари з БД
    docs = asyncio.run(load_products_from_db())
    vectorStore = create_db(docs=docs)
    retriever = vectorStore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="product_search",
        description="Use this tool to search for phone accessories, cases, and Apple products from our store"
    )

    tools = [retriever_tool]

    agent_obj = create_openai_functions_agent(
        llm=llm_obj,
        prompt=prompt_obj,
        tools=tools
    )

    agentExecutor = AgentExecutor(
        agent=agent_obj,
        tools=tools,
        memory=memory,  # ✅ Add memory here
        verbose=True
    )
    return agentExecutor



if __name__ == "__main__":
    session_id = "default_user"  # You can customize this or ask the user
    agent = create_agent(session_id=session_id)

    while True:
        user_input = str(input("Input: \n"))
        if user_input.lower() == "exit":
            break

        response = agent.invoke({"input": user_input})
        print(f"Output: \n {response.get('output')}")
