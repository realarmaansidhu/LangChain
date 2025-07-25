# LangChain CommaSeparatedOutputParser Example in Groq LLM
import os
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq as Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9)

parser = CommaSeparatedListOutputParser()

prompt = PromptTemplate(
    template=(
        "List five famous authors related to the {genre}.\n"
        "Return ONLY the names, separated by commas. Do not include numbers, explanations, or any extra text."
    ),
    input_variables=["genre"],
)

chain = prompt | llm | parser
response = chain.invoke({"genre": "science fiction"})
print(response)