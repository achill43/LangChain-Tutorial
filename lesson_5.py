# Adding Chat History to Chatbot
import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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


def create_chat():
    docs = get_document_from_web(
        url="https://byte93.pythonanywhere.com/articles/articles/biblioteka-asyncio"
    )
    vectorStore = create_db(docs=docs)
    prompt_obj = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    chain_obj = create_stuff_documents_chain(llm=llm, prompt=prompt_obj)

    retrieval = vectorStore.as_retriever()
    retrieval_chain = create_retrieval_chain(retrieval, chain_obj)

    return retrieval_chain




if __name__ == "__main__":
    chat = create_chat()
    chat_history = [
        HumanMessage(content="Привіт"),
        AIMessage(content="Привіт. Чим я можу вам допомогти?")
    ]
    while True:
        user_input = str(input("Your question: \n"))    # Що таке Asyncio?
        if user_input.lower() == "exit":
            break
        response = chat.invoke({"input": user_input, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.get("answer")))
        print("Answer:")
        print(response.get("answer"))
