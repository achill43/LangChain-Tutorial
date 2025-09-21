# pip install langchain-openai pydantic
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("API key not found. Make sure .env file is set correctly.")

# Initialize a single ChatOpenAI instance for simplicity
chat = ChatOpenAI(
    api_key=OPENAI_API_KEY,  # Use the API key directly
    temperature=0.7,
    max_tokens=500,
    model="gpt-4-1106-preview",
)


# Define the Pydantic model directly from pydantic library
class WeatherReport(BaseModel):
    city: str = Field(description="City name")
    report: str = Field(description="Brief weather report")


# General prompt for the task
prompt_text = """Write a weather report for a major city
in ten words or less. Do not include any additional explanation."""


def baseline():
    """Demonstrates a simple baseline call without guided output."""
    print("Baseline:")
    print(chat.invoke(prompt_text).content)


def with_guided_prompt():
    """Demonstrates a simple, manually guided prompt."""
    print("\n1. Ask nicely")
    guided_prompt = (
        prompt_text
        + """
Return the result as JSON as follows:
{ "city": "<CITY_NAME>",
"report": "<BRIEF_REPORT>" }
"""
    )
    print(chat.invoke(guided_prompt).content)


def with_pydantic_output_formatter():
    """
    Uses the recommended JsonOutputParser to guide the LLM's output
    and parse it into a Pydantic object.
    """
    print("\n2. JsonOutputParser (Recommended)")

    # The modern way to create a parser and get instructions
    parser = JsonOutputParser(pydantic_object=WeatherReport)

    # Create the runnable prompt with the instructions in a SystemMessage
    runnable_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=parser.get_format_instructions()),
            HumanMessage(content="{prompt_text}"),
        ]
    )

    # Create the chain using LCEL
    chain = runnable_prompt | chat | parser

    # Invoke the chain. The input dictionary key must match the prompt variable.
    py_obj = chain.invoke({"prompt_text": prompt_text})

    # Access the attributes of the parsed Pydantic object
    city = py_obj.get("city")
    report = py_obj.get("report")
    print(f"City: {city}")
    print(f"Report: {report}")


if __name__ == "__main__":
    baseline()
    with_guided_prompt()
    with_pydantic_output_formatter()
