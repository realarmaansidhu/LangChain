# Langchain code to implement a Semantic Similarity Example Selector using FewShotPromptTemplate and Deeplake VectorStore
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import DeepLake
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model_name="llama-3.3-70b-versatile", temperature=0.9)

# Create a Deep Lake dataset
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
my_activeloop_org_id = "armaansidhu" 
my_activeloop_dataset_name = "langchain_course_fewshot_selector"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)

temp = input("Enter a temperature in Celsius (e.g., 25): ").strip()

examples = [
    {"input": "0°C", "output": "32°F"},
    {"input": "10°C", "output": "50°F"},
    {"input": "20°C", "output": "68°F"},
    {"input": "30°C", "output": "86°F"},
    {"input": "40°C", "output": "104°F"},
]

example_prompt = PromptTemplate(input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

prefix = "Here are some examples of temperature conversions:\n"

suffix = (
    "\nGiven the above examples, convert the temperature below from Celsius to Fahrenheit."
    "\nInput: {temp}\nOutput (just the number and °F, nothing else):"
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,           # our list of example dicts
    embeddings,         # our embedding function
    db,                 # our DeepLake vectorstore
    k=1
)

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    prefix=prefix,
    suffix=suffix,
    input_variables=["temp"],
    example_separator="\n"
)

chain = prompt | llm
chain.invoke({"temp": temp})
# Print the response
response = chain.invoke({"temp": temp})
print("Converted Temperature:", response.content.strip().splitlines()[0])