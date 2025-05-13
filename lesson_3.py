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
    # Create Prompt Template
    prompt_1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate a list of 10 synonyms for the following word. Return the result as a comma seperated list",
            ),
            ("human", "{input}"),
        ]
    )

    parser = StrOutputParser()

    # Create LLM Chain
    chain_1 = prompt_1 | llm | parser

    response = chain_1.invoke({"input": "colorful"})

    return response


def call_list_output_parser():
    # Create Prompt Template
    prompt_1 = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate a list of 10 synonyms for the following word. Return the result as a comma seperated list",
            ),
            ("human", "{input}"),
        ]
    )

    parser = CommaSeparatedListOutputParser()

    # Create LLM Chain
    chain_1 = prompt_1 | llm | parser

    response = chain_1.invoke({"input": "colorful"})

    return response


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


# content = call_string_output_parser()
# content = call_list_output_parser()
content = call_json_output_parser()
print(f"{content=}")
