# Prompt templates and Chain
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
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
)


# # Create Prompt Template
# prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}")


# # Create LLM Chain
# chain_one = prompt | llm


# response = chain_one.invoke({"subject": "dog"})

# print(f"{response=}")

prompt_2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate a list of 10 synonyms for the following word. Return the result as a comma seperated list",
        ),
        ("human", "{input}"),
    ]
)

chain_2 = prompt_2 | llm


response = chain_2.invoke({"input": "collorfull"})

print(f"{response=}")
