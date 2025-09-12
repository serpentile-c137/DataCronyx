from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew

load_dotenv()

llm = LLM(
    model = "gemini/gemini-2.0-flash",
    temperature = 0.7,
)

result = llm.call("What is capital of India?")
print(result)