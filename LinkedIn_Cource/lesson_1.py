import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()


def main(api_key: str):
    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-5",
    )

    system_prompt = """Given the following short description
        of a particular topic, write 3 attention-grabbing headlines 
        for a blog post. Reply with only the titles, one on each line,
        with no additional text.
        DESCRIPTION:
    """
    user_input = """AI Orchestration with LangChain and LlamaIndex
        keywords: Generative AI, applications, LLM, chatbot"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]

    # Use the invoke method to call the LLM
    response = llm.invoke(messages)

    # Print the content of the response
    print(response.content)


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("API key not found. Make sure .env file is set correctly.")
    else:
        main(api_key=api_key)
