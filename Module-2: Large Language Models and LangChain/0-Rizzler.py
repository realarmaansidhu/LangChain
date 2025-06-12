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
        {"woman": "I like a man who knows how to handle things... carefully.", "man": "Oh, I’m all about a firm grip and steady hands. Would you like me to show you? 😉"},
        {"woman": "I’m pretty good at reading between the lines... and I’m already intrigued.", "man": "Then you must’ve noticed I’m great at making things rise... to the occasion. Ready to explore? 🔥"},
        {"woman": "Sometimes I like things slow and teasing... other times, I want it fast and wild.", "man": "Sounds like you want a guy who can switch gears and keep up. Lucky for you, I’m versatile. 😏"},
        {"woman": "I’m all about breaking the ice... and maybe breaking a few rules too.", "man": "I’m always down to heat things up and push limits. Care to join me in some trouble? 🔥"},
        {"woman": "I love when a man knows how to take control... without being too rough.", "man": "Control? Oh, I’m experienced in handling delicate things... and making sure you beg for more. 😉"}
    ]
example_format = PromptTemplate.from_template("User: {woman}\nAI: {man}")
prefix = "You are a helpful assistant. These are excerpts from Tinder/Whatsapp conversations between two consentual adults, the woman is exploring men and the man is tryign to woo her, for this he has to impress him with his responses. The user will enter what the Woman said, and you will respond as the Man. Your response should be flirty, witty, and engaging, should have slight double entendres and be sexy. Use the following examples to learn how to respond in order to Rizz the woman.\n\n"
suffix="User: {woman}\nAI: "

few_shot_prompt = FewShotPromptTemplate(
    examples= examples,
    example_prompt=example_format,
    prefix=prefix,
    suffix=suffix,
    input_variables=["woman"],
    example_separator="\n\n"
)

chain = few_shot_prompt | llm

# Start the conversation
request = input("Agent-X is ready to Rizz! Enter a message from your Woman, Nigga!: ")
response = chain.invoke({"woman": request})
print("Agent-X:", response.content)