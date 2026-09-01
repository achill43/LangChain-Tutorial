import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Make sure .env file is set correctly.")

# Initialize model
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.7,
)

response = llm.invoke("Hello, tall me: What is LLM inside Artificial intelligence?")


print(f"{response=}")


response = llm.batch(
    [
        "Hello, tall me: What is LLM inside Artificial intelligence?",
        "What is a LangChain?",
    ]
)

for batch in response:
    print(f"{batch.content=}")
