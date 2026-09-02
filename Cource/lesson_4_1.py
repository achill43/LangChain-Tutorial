# RAG system tutorial
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Make sure .env file is set correctly.")

model = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)

document_a = Document(page_content="""
    Модуль threading у Python дозволяє запускати кілька потоків (менших одиниць процесу) паралельно. Це чудово підходить для завдань, пов’язаних із 
    вводом-виводом, таких як мережеві запити чи операції з файлами, але не ідеально підходить для завдань, пов’язаних із CPU (через Global Interpreter Lock (GIL)).
    Threading - це модуль, який дозволяє створювати та керувати потоками у Python. Потоки - це легкі процеси, які можуть виконуватися паралельно в 
    межах одного процесу. Це корисно для виконання кількох завдань одночасно, таких як обробка запитів або виконання асинхронних операцій.
    """)

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

question = "Що таке Threading?"

print(f"{question=}")

response = chain.invoke({"question": question, "context": [document_a]})

print(f"{response=}")
