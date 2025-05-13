# Documents using Retrieval Chains
import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic.types import SecretStr


# Load environment variables from .env file
load_dotenv()


# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")


if not api_key:
    raise ValueError("API key not found. Make sure .env file is set correctly.")


# Initialize ChatOpenAI
llm = ChatOpenAI(
    api_key=SecretStr(api_key),
    model="gpt-3.5-turbo",
    temperature=0.7,
)

# Create Document
docA = Document(
    page_content="""Serhii: Python is a computer programming language often used to build websites and software,
    automate tasks, and analyze data. Python is a general-purpose language, not specialized for any
    specific problems, and used to create various programmes"""
)

# prompt = ChatPromptTemplate.from_template(
#     """
# Answers the user's question:
# Context: {context}
# Question: {question}
# """
# )

# chain_1 = prompt | llm


# response = chain_1.invoke({
#     "question": "What is Python?",
#     "context": [docA]
# })

# print(f"{response.content=}")


def get_document_from_web(url: str) -> list[Document]:
    loader = WebBaseLoader(url)
    docs = loader.load()
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=20
    )
    split_docs = spliter.split_documents(docs)
    return split_docs


def create_db(docs: list[Document]):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


docs = get_document_from_web(url="https://www.coursera.org/in/articles/what-is-python-used-for-a-beginners-guide-to-using-python")
vectorStore = create_db(docs=docs)

prompt_2 = ChatPromptTemplate.from_template(
    """
Answers the user's question:
Context: {context}
Question: {input}
"""
)

chain_2 = create_stuff_documents_chain(
    llm=llm, prompt=prompt_2
)

retrieval = vectorStore.as_retriever()
retrieval_chain = create_retrieval_chain(retrieval, chain_2)

response = retrieval_chain.invoke({
    "input": "What is Python?"
})

print(f"{response=}")
