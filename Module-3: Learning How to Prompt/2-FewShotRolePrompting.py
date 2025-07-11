# Langchain code to illustrate few-shot role prompting with Groq LLM
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9)

# Define a few-shot prompt template
examples = [
    {"country": "India", "city": "Bombay"},
    {"country": "USA", "city": "New York City"},
    {"country": "Pakistan", "city": "Karachi"},
    {"country": "Canada", "city": "Toronto"},
    {"country": "Australia", "city": "Sydney"},
    {"country": "Germany", "city": "Frankfurt"},
    {"country": "China", "city": "Shanghai"},
]

prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_separator="\n",
    example_prompt= PromptTemplate(
        input_variables=["country", "city"],
        template="Country: {country}\nCity: {city}",
    ),
    prefix="You are a helpful assistant that provides a city for each country.\n",
    suffix="Now study the examples above, to suggest a city of similar relevance for - \nCountry: {country}\nCity:\n(print just the name, nothing else)",
    input_variables=["country"]
)

user_input = input("Enter a country to suggest a city: ").strip()
prompt = prompt_template.format(country=user_input)
response = llm.invoke(prompt)
print(response.content.strip())