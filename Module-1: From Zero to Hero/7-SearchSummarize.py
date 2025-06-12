#We will make a langchain agent that can search the web and summarize the results. It will have two different tools for this, one to search another to summarize. We will use the Groq LLM to do this.
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv(dotenv_path=".env")  # This loads variables from .env into the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
print("Loaded API key:", GROQ_API_KEY)
print("Loaded Google Search API key:", GOOGLE_SEARCH_API_KEY)
print("Loaded Google CSE ID:", GOOGLE_CSE_ID)

# Initialize the Google Search API wrapper
llm = Groq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9
)

request = input("What would you like to present the summarized search for? ")

#First, we will create a prompt template for summarization
prompt = PromptTemplate(
    input_variables=["request"],
    template="Summarize the following text: {request}"
)
summarize_chain = LLMChain(llm=llm, prompt=prompt)

#Then, we will create a tool to search the web using the Google Search API
search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_SEARCH_API_KEY, google_cse_id=GOOGLE_CSE_ID)
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Use this tool to search the web for information. Input should be a search query."
    ),
    Tool(
        name="Summarize",
        func=LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                input_variables=["text"],
                template="Summarize the following text: {text}"
            )
        ).run,
        description="Use this tool to summarize the results of a web search. Input should be the text to summarize."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  
)

response = agent({"input": request})
print(response['output'])