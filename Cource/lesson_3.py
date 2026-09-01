# Output Parsers (String, List, JSON)
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

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


def call_string_output_parser():
    # Create prompt for our LLM
    prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}.")

    # Create output parser
    parser = StrOutputParser()

    # Create LLM chain
    chain = prompt | llm | parser

    # Make query for our LLM
    response = chain.invoke({"subject": "dog"})
    return response


response = call_string_output_parser()
print(f"{response=}")


def call_list_ouput_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI cheff. Create a unique recipe based on the follow main ingredient. Always give answers on Ukrainian language",
            ),
            ("human", "{input}"),
        ]
    )

    # Create list optput parser
    parser = CommaSeparatedListOutputParser()

    # Create LLM chain
    chain = prompt | llm | parser

    # Make query for our LLM
    response = chain.invoke({"input": "cheese"})
    return response


response = call_list_ouput_parser()
print(f"{response=}")


def call_json_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract a JSON object with `name`, `age`, and `position` from the text.",
            ),
            ("human", "{input}"),
        ]
    )

    class Person(BaseModel):
        name: str
        age: int
        position: str

    parser = JsonOutputParser(pydantic_object=Person)

    chain = prompt | llm | parser

    response = chain.invoke(
        {"input": "Serhii is 32 years old and working as a programmer"}
    )
    return response


print(call_json_output_parser())
