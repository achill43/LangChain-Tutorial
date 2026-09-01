# Prompt templates and Chain
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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


# Create prompt for our LLM
prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}.")

# Create LLM chain

chain = prompt | llm

# Make query for our LLM
response = chain.invoke({"subject": "dog"})

print(f"{response=}")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI cheff. Create a unique recipe based on the follow main ingredient. Always give answers on Ukrainian language",
        ),
        ("human", "{input}"),
    ]
)

# Create LLM chain

chain = prompt | llm

# Make query for our LLM
response = chain.invoke({"input": "cheese"})

print(f"{response=}")
