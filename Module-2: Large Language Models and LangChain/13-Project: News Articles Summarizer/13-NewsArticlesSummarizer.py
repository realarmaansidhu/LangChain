# Langchain code to implement an Online News Articles Summarizer that dynamically fetches and summarizes news articles using the Groq LLM.

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as Groq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(dotenv_path=".env")
print("Loaded API key:", os.getenv("GROQ_API_KEY"))

import requests
from newspaper import Article

# fetch news articles from a URL
import requests
from newspaper import Article

def fetch_news_article(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        session = requests.Session()
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            article = Article(url)
            article.set_html(response.text)
            article.parse()
            return article.text, article.title
        else:
            print(f"Failed to fetch article at {url} (status code: {response.status_code})")
            return None, None
    except Exception as e:
        print(f"Error occurred while fetching article at {url}: {e}")
        return None, None

llm = Groq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.9
)

url = input("Enter a news article URL: ")
article_text, article_title = fetch_news_article(url)

if (article_text, article_title):
    system_message = SystemMessage(content="You are a very good assistant that summarizes online articles.")
    template = """
    Here's the article you want to summarize.

    ==================
    Title: {article_title}

    {article_text}
    ==================

    Write a summary of the previous article.
    """
    prompt = template.format(
        article_title=article_title, article_text=article_text
    )
    human_message = HumanMessage(content=prompt)
    messages = [system_message, human_message]
    response = llm.invoke(messages)
    print(response.content)
else:
    print("Failed to fetch the article. Please check the URL and try again.")
    exit()