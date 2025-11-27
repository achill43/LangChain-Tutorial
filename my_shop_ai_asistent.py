# Long term SQLite Agent memory
import os
import asyncio
import sys

from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langgraph.prebuilt import create_react_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic.types import SecretStr

from sqlalchemy import text
from docs_db_connection import AsyncSessionLocal


async def load_products_from_db() -> list[Document]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("SELECT model, name, description, price FROM products")
        )
        rows = result.fetchall()

        docs = []
        for row in rows:
            content = f"Модель: {row.model}\nНазва: {row.name}\nОпис: {row.description}\nЦіна: {row.price} грн"
            doc = Document(page_content=content, metadata={"source": "products_db"})
            docs.append(doc)
        print("Loaded products from DB:")
        print(docs)
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


def create_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///agent_memory.db",
    )


def create_agent(session_id: str):
    # Load products and create retriever
    docs = asyncio.run(load_products_from_db())
    vectorStore = create_db(docs=docs)
    retriever = vectorStore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="product_search",
        description="Use this tool to search for phone accessories, cases, and Apple products.",
    )

    tools = [retriever_tool]

    # ⭐ System message goes here — REACT ignores custom prompts otherwise
    system_prompt = "You are an AI sales consultant in an online store. After receiving a request from a user about the characteristics of the product they are interested in, find the most suitable product in your database of our store and offer it to the user."
    # 1. Define the Prompt Template
    # This is the standard structure for LangChain/LangGraph agents
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            # MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Create REACT agent (LangGraph)
    agent = create_react_agent(
        model=llm_obj,
        tools=tools,
        prompt=prompt_template,
    )

    # ⭐ Wrap agent with message history
    agent_with_memory = RunnableWithMessageHistory(
        agent,
        create_history,
        input_messages_key="input",  # this MUST be "input"
        history_messages_key="history",  # this MUST be "history"
    )

    return agent_with_memory


if __name__ == "__main__":
    session_id = "default_user"  # You can customize this or ask the user
    agent = create_agent(session_id=session_id)

    while True:
        print("Введіть ваше запитання: \n")
        # Hi I need metal case for iPhone 15 Pro Max with a card holder
        user_input_bytes = sys.stdin.buffer.readline()
        user_input = user_input_bytes.decode("utf-8").strip()
        if user_input.lower() == "exit":
            break

        response = agent.invoke(
            {"input": user_input}, config={"configurable": {"session_id": session_id}}
        )
        final_message = response["messages"][-1]
        final_output_text = final_message.content

        print(f"Output: \n {final_output_text}")
