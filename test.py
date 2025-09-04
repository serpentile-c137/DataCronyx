import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()  # Loads variables from .env into environment

# Set your Google API key here or via environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

# Use the correct Gemini model name (e.g., "gemini-1.5-flash" or "gemini-pro")
model = genai.GenerativeModel("gemini-2.5-flash")
response = model.generate_content("What is the capital of France?")
print(response.text)
