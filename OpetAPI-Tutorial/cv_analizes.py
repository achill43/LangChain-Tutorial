import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from pprint import pprint

load_dotenv()


def get_text_from_pdf(file_path):
    # Open the PDF file

    # Initialize a variable to store text
    text = ""

    with open(file_path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Iterate through all the pages
        for page in pdf_reader.pages:
            # Extract text from each page
            text += page.extract_text()

    return text


context = get_text_from_pdf("OpetAPI-Tutorial/cv.pdf")

print("Analize your CV is in process...")

prompt = f"""PDF Content:
{context}

Respond only with valid JSON. Example: 
{{"full_name": "John Brown", "position": "Software Engineer", "work_experiance": "1 year", "companies": ["Company 1", "Company 2"], "skills": {{"category 1": ["Skill 1", "Skill 2", "Skill 3"], "category 2": ["Skill 4", "Skill 5", "Skill 6"]}}}}.
Where:
- full_name is your suggestion how must cold this text
- position is the main information from this text max length 100 worlds.
- work_experiance is amount of work experiance years from this text
- companies is a list of company names from this text
- skills is a dictionary of skills from this text splited by categories"""

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "low"},
    input=prompt,
)

responce_dict = json.loads(response.output_text)
pprint(responce_dict)
