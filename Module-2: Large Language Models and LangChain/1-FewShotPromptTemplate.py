# We will create a few-shot prompt template that can be used to generate responses based on a few examples. This is useful for tasks where you want the model to learn from a few examples and then apply that knowledge to new inputs.
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
load_dotenv(dotenv_path=".env")  # This loads variables from .env into the environment
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9
)
# Define a few-shot prompt template
examples = [
        {
        "query": "What's the weather like?",
        "answer": "It's raining cats and dogs, better bring an umbrella!"
    }, {
        "query": "How old are you?",
        "answer": "Age is just a number, but I'm timeless."
    }
    ]
example_format = PromptTemplate.from_template("User: {query}\nAI: {answer}")
prefix = "The following are excerpts from conversations with an AI assistant. The assistant is known for its humor and wit, providing entertaining and amusing responses to users' questions. Here are some examples:"
suffix="User: {query}\nAI: "

few_shot_prompt = FewShotPromptTemplate(
    examples= examples,
    example_prompt=example_format,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

chain = few_shot_prompt | llm

# Start the conversation
request = input("Enter a message: ")
response = chain.invoke({"query": request})
print("", response.content)