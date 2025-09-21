# Long term SQLite Agent memory
import os

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
from langchain_community.document_loaders import WebBaseLoader, TextLoader
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


def get_documents(source_path: str, is_file: bool = False) -> list[Document]:
    """
    Loads documents from either a web URL or a local file.
    Args:
        source_path: The URL or file path.
        is_file: Set to True if the source_path is a local file.
    Returns:
        A list of LangChain Document objects.
    """
    if is_file:
        loader = TextLoader(file_path=source_path)
    else:
        loader = WebBaseLoader(web_path=source_path)

    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    return split_docs


def create_db(docs: list[Document]):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def create_agent(session_id: str):
    # Create message history using SQLite
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///agent_memory.db",  # Replace with your desired database path
    )

    # Wrap it in buffer memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, chat_memory=history
    )

    prompt_obj = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly assistant called Max witch given information in Python program language area.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    file_path = "LinkedIn_Cource/answers.txt"
    docs = get_documents(source_path=file_path, is_file=True)
    vectorStore = create_db(docs=docs)
    retriever = vectorStore.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="answer_search",
        description="Use this tool to search for answers from the documents",
    )

    tools = [retriever_tool]

    agent_obj = create_openai_functions_agent(
        llm=llm_obj, prompt=prompt_obj, tools=tools
    )

    agentExecutor = AgentExecutor(
        agent=agent_obj, tools=tools, memory=memory, verbose=True  # ✅ Add memory here
    )
    return agentExecutor


if __name__ == "__main__":
    session_id = "develper_user"  # You can customize this or ask the user
    agent = create_agent(session_id=session_id)

    while True:
        user_input = str(input("Input: \n"))
        if user_input.lower() == "exit":
            break

        response = agent.invoke({"input": user_input})
        print(f"Output: \n {response.get('output')}")
