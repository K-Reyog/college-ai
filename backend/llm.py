import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set. Add it to backend/.env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


def ask_gemini(question):
    """Sends a question to Gemini and returns the response text."""
    try:
        response = model.generate_content(question)
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}") from e

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text
