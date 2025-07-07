# Langchain code to show use case of ChatPromptTemplate, SystemMessagePromptTemplate, and HumanMessagePromptTemplate using a movie summary example

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate 

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

movie = input("Enter any Movie whose Summary you want to know: ")

sys_msg_prompt = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant that provides movie summaries."
)

hmn_msg_prompt = HumanMessagePromptTemplate.from_template(
    "Please summarize the movie: {movie}"
)

prompt = ChatPromptTemplate.from_messages(
    [sys_msg_prompt, hmn_msg_prompt]
)

llm = Groq(model_name="llama-3.3-70b-versatile", temperature=0.1)

prompt_chain = prompt | llm
response = prompt_chain.invoke({"movie": movie})
print("Summary:", response.content.strip()) 