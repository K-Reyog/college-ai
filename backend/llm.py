import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load variables from .env
load_dotenv()

# Configure Gemini with your API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")


def ask_gemini(question):
    """
    Sends a question to Gemini and returns the response text.
    """

    response = model.generate_content(question)

    return response.text