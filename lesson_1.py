import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("API key not found. Make sure .env file is set correctly.")

# Initialize ChatOpenAI
llm = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000,
)

# response = llm.invoke("Hello, tall me what is LLM?")
response = llm.batch(["Hello, tall me what is LLM?", "What is LangChain?"])


print(f"{response=}")
for batch in response:
    print(f"{batch.content=}")

# response = llm.stream("Hello, tall me what is LLM?")

# for chunk in response:
#     print(chunk.content, end="", flush=True)
