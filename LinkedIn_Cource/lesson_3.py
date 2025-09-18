# Local Documents
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Load environment variables from .env file
load_dotenv()


def main(api_key: str, user_input: str):
    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-5",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant that give. information about writer bio",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"input": user_input})

    # Print the content of the response
    print(response)


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Make sure .env file is set correctly.")
    else:

        user_input = """William Shakespeare, keywords: writer, bio"""
        main(api_key=api_key, user_input=user_input)
