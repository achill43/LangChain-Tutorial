# RAG system tutorial Vector database and Embedings
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Make sure .env file is set correctly.")


def get_document_from_url(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Number of split documents: {len(split_docs)}")
    return split_docs


def create_vector_database(documents):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


document_a = get_document_from_url(
    "https://byte93.pythonanywhere.com/articles/articles/threading"
)

print(f"{document_a=}")

vector_database = create_vector_database(document_a)


def create_chain(vector_database):
    model = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
    # create_retrieval_chain feeds the question in under the key "input",
    # and fills "context" itself with whatever the retriever returns.
    prompt = ChatPromptTemplate.from_template("""
        Answer the users's question:
        Context: {context}
        Question: {input}
        """)

    # Create output parser
    parser = StrOutputParser()
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt,
        document_variable_name="context",
        output_parser=parser,
    )
    # Create a chain that retrieves relevant documents from the vector database and then uses the chain to answer the question
    retriever = vector_database.as_retriever()
    # retrieval_chain = retriever | chain
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=chain,
    )
    return retrieval_chain


question = "Що таке Threading?"

print(f"{question=}")

chain = create_chain(vector_database)

# Only the question goes in: the retriever picks the relevant chunks out of the
# vector database, so passing the whole document would defeat the point.
response = chain.invoke({"input": question})

# The result is a dict with "input", "context" (the retrieved chunks) and "answer".
print(f"Retrieved {len(response['context'])} chunk(s) from the vector database")
print(f"answer={response['answer']}")
