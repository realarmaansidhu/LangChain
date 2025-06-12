import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="/.env")
google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY")
google_cse_id=os.getenv("GOOGLE_CSE_ID")

print("Loaded Google API key:", google_api_key)
print("Loaded Google CSE ID:", google_cse_id)

from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper(
    google_api_key=google_api_key,
    google_cse_id=google_cse_id
)

tool = Tool(
    name="google_search",
    description="Search Google for recent results.",
    func=search.run,
)

# search_results = tool.invoke("Obama's first name?")
# # Wrap the search result in an LLm to extract the first name
# print("Search Results: ", search_results)

# from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq as Groq
llm = Groq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)
# prompt = PromptTemplate(
#     input_variables=["search_results"],
#     template="Extract the first name from the following search results: {search_results}"
# )
# chain = prompt | llm
# response = chain.invoke({"search_results": search_results})
# print("Extracted First Name: ", response.content.strip())

#Agentic Run
from langchain.agents import initialize_agent, AgentType
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Useful for searching the web for recent information."
    ),
]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

request = input("Ask a question to the Agentic AI System: ")
response = agent.invoke(request)
print("Agent Response: ", response['output'].strip())
# The agent should be able to answer the question using the search results.
# Note: Make sure to set the GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID in your .env file. 