# Langchain code to implement an Example Selector for a few shot prompting example
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

llm = Groq(model_name="llama-3.3-70b-versatile", temperature=0.0)

# Define a few-shot prompt template
examples = [
    {"year": "2000", "conflict": "Second Chechen War (Russia vs. Chechen separatists)"},
    {"year": "2001", "conflict": "U.S. invasion of Afghanistan after 9/11 attacks"},
    {"year": "2002", "conflict": "Afghanistan War – U.S. combats Taliban and Al-Qaeda insurgents"},
    {"year": "2003", "conflict": "U.S.-led invasion of Iraq and the beginning of the Iraq War"},
    {"year": "2004", "conflict": "Second Battle of Fallujah (Iraq) – intense urban warfare"},
    {"year": "2005", "conflict": "Escalation of Iraq insurgency and sectarian violence"},
    {"year": "2006", "conflict": "Israel–Hezbollah War (Lebanon)"},
    {"year": "2007", "conflict": "Surge in Iraq – heavy U.S. troop deployment and violence"},
    {"year": "2008", "conflict": "Russo-Georgian War over South Ossetia"},
    {"year": "2009", "conflict": "Sri Lankan Civil War ends – deadly final offensive against Tamil Tigers"},
    {"year": "2010", "conflict": "Mexican Drug War escalates – military confrontations intensify"},
    {"year": "2011", "conflict": "Syrian Civil War begins with brutal Assad regime crackdowns"},
    {"year": "2012", "conflict": "Syrian Civil War escalates with urban battles and chemical attacks"},
    {"year": "2013", "conflict": "Battle of Qusayr (Syria) – key turning point with heavy casualties"},
    {"year": "2014", "conflict": "War in Donbas (Ukraine) begins after Russian annexation of Crimea"},
    {"year": "2015", "conflict": "Yemen Civil War escalates with Saudi-led coalition airstrikes"},
    {"year": "2016", "conflict": "Battle of Aleppo (Syria) – one of the deadliest urban battles"},
    {"year": "2017", "conflict": "Battle of Mosul (Iraq) – brutal anti-ISIS operation"},
    {"year": "2018", "conflict": "Ongoing Yemen conflict – port battles and humanitarian crisis"},
    {"year": "2019", "conflict": "Libyan Civil War escalates – Tripoli offensive by Haftar's forces"},
    {"year": "2020", "conflict": "Second Nagorno-Karabakh War (Armenia vs. Azerbaijan)"},
    {"year": "2021", "conflict": "Afghanistan – Taliban regains power after U.S. withdrawal"},
    {"year": "2022", "conflict": "Russian invasion of Ukraine – full-scale war begins"},
    {"year": "2023", "conflict": "Bakhmut and Avdiivka battles – some of the bloodiest in Ukraine"},
    {"year": "2024", "conflict": "Gaza–Israel War – heavy civilian toll and regional escalation"},
    {"year": "2025", "conflict": "Ongoing Ukraine conflict with intensified drone and missile strikes"},
]

example_prompt = PromptTemplate(
    input_variables=["year", "conflict"],
    template="{year}: {conflict}"
)

prefix = "Here are some examples of years and something related to that year:\n"

suffix = "\nGiven the above examples, print ONLY the one-line event name for the year below. If the year is > 2025 but < 2030, predict an event seeing the examples and current affairs. If {year}>2030, just say 'Unknown':\n{year}: "

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=100  
)

prompt = FewShotPromptTemplate(
    # examples=examples, # Uncomment this line and comment the example_selector if you want to use static examples
    example_prompt=example_prompt,
    example_selector=example_selector,
    prefix=prefix,
    suffix=suffix,
    input_variables=["year"],
    example_separator="\n"
)

year = input("Enter a year: ").strip()

chain = prompt | llm
response = chain.invoke({
    "year": year})
print(response.content.strip())