# Langchain Pydantic Output Parser Example with Groq LLM
import os
from langchain_groq import ChatGroq as Groq
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model="llama-3.3-70b-versatile", temperature=0.9)

class Suggestions(BaseModel):
    suggestions: list[str] = Field(
        description="List of suggestions for social media usernames"
    )

    @field_validator("suggestions", mode="before")
    def validate_suggestions(cls, v):
        # v is the whole list before parsing
        if not isinstance(v, list):
            raise ValueError("Suggestions must be a list.")
        validated = []
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Each suggestion must be non-empty.")
            if not item[0].isalpha():
                raise ValueError("Each suggestion must start with a letter.")
            validated.append(item.strip())
        return validated
    
parser= PydanticOutputParser(pydantic_object=Suggestions)

prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "You are an expert at creating catchy social media usernames.\n"
        "Generate 10 creative usernames for a profile about {topic}.\n"
        "Return ONLY a valid JSON object with the following format:\n"
        "{{\n"
        '  "suggestions": ["username1", "username2", ...]\n'
        "}}\n"
        "Do not include any explanations, markdown, or extra text. Only output the JSON object."
    )
)

chain =  prompt | llm | parser
response = chain.invoke({"topic": "travel and photography"})
print("-" * 50 + "\n")
for suggestion in response.suggestions:
    print(suggestion)
print("-" * 50 + "\n")