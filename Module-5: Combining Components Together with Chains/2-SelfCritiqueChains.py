# Rewritten LangChain code to execute an LLM Chain with a Constitutional Chain for self-critique.

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

# Initialize the LLM
llm = Groq(model="llama-3.3-70b-versatile", temperature=0.0)

# Define the ethical principle for self-critique
ethics = ConstitutionalPrinciple(
    name="Ethics",
    critique_request="The output should be fair and just. No harmful advice should ever be given to the users.",
    revision_request="If the output is harmful or unfair, it should be revised."
)

fun = ConstitutionalPrinciple(
    name="Fun",
    critique_request="The output should be fun and sarcastic. It should be easy to grab by a 7th grader.",
    revision_request="If the output is unfunny or unsarcastic, it should be revised."
)

# Prompt for generating “evil” advice
evil_prompt = PromptTemplate(
    template="You are an evil assistant. For the following user input, suggest awful ideas that will negatively affect the user. Print only the advice directly:\n{user_input}",
    input_variables=["user_input"]
)

funny_prompt = PromptTemplate(
    template="You are a fun and sarcastic assistant. Make the following advice funny, witty, and safe for a 7th grader. Print only the advice directly:\n{advice}",
    input_variables=["advice"]
)

llm_chain_evil = LLMChain(
    llm=llm,
    prompt=evil_prompt,
    output_key="advice"
)

llm_chain_funny = LLMChain(
    llm=llm,
    prompt=funny_prompt,
    output_key="advice"
)

self_ethics = ConstitutionalChain.from_llm(
    chain=llm_chain_evil,
    constitutional_principles=[ethics],
    llm=llm,
    verbose=True
)

self_fun = ConstitutionalChain.from_llm(
    chain=llm_chain_funny,
    constitutional_principles=[fun],
    llm=llm,
    verbose=True
)

# Get user input
user_input = input("Enter your Question: ")

# Step 1: Generate original "evil" advice
original_advice = llm_chain_evil.run({"user_input": user_input})
print("\nOriginal Advice (unsafe):")
print(original_advice)

# Step 2: Pass the original advice through the ethical ConstitutionalChain
revised_ethical_advice = self_ethics.run({"user_input": user_input})
print("\nRevised Advice (ethical):")
print(revised_ethical_advice)

# Step 3: Pass the revised ethical advice through the fun ConstitutionalChain
revised_fun_advice = self_fun.run({"advice": revised_ethical_advice})
print("\nRevised Advice (fun):")
print(revised_fun_advice)