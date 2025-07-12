# Langchain code to illustrate simple chain prompting
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import PromptTemplate

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.0)

country = input("Enter a country to suggest a scientist: ").strip()

question = "What is the name of the Scientist considered the father of {country} Nuclear Program?"

prompt = PromptTemplate(
    input_variables=["country", "template_question"],
    template= """
    For the below question, answer ONLY with the Scientist's name.
    Do NOT print anything else, not even a period or explanation.
    
    {question}

    If the {country} doesn't have such a scientist, print exactly: Unknown.
    """
)

chain = prompt | llm
response = chain.invoke({"country": country, "question": question.format(country=country)})
scientist = response.content.strip()
print(scientist)
print("-" * 50 + "\n")

if scientist == "Unknown":
    print("No relevant Nuclear Scientist found.")
else:
    prompt = PromptTemplate(
        input_variables = ["scientist"],
        template = """
        For the below Scientist, provide a very brief summary of their life and major contributions to science. Provide timelines and key events. Don't use **s for bold italics etc.\n
        {scientist}
        """
    )
    chain = prompt | llm
    response = chain.invoke({"scientist": scientist})
    print(response.content.strip())