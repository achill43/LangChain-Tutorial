# RAG system tutorial Document spliter
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Make sure .env file is set correctly.")

model = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)


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


prompt = ChatPromptTemplate.from_template("""
    Answer the users's question:
    Context: {context}
    Question: {question}
    """)

# Create output parser
parser = StrOutputParser()

# chain = prompt | model | parser
chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
    document_variable_name="context",
    output_parser=parser,
)

document_a = get_document_from_url(
    "https://byte93.pythonanywhere.com/articles/articles/threading"
)

print(f"{document_a=}")

question = "Що таке Threading?"

print(f"{question=}")

response = chain.invoke({"question": question, "context": document_a})

print(f"{response=}")
