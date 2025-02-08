#os.environ[LANGCHAIN_API_KEY]="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"
LANGCHAIN_PROJECT="chatbot"

# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()
import os

# Replace 'your_api_key_here' with the actual API key
os.environ["LANGCHAIN_API_KEY"] = "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"

#langsmith tracking

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
import os
print(os.getenv("LANGCHAIN_API_KEY"))
#creating chatbot

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Please provide response to the user queries"),
        ("user","Question:{question}")
     ]
)
#streamlit framework 
st.title("Chatbot try")
input_text = st.text_input("Enter your question")

#llm call
llm=Ollama(model="llama2")
output_parsers=StrOutputParser()

#chain 
chain=prompt|llm|output_parsers

if input_text:
  st.write(chain.invoke({"question":input_text}))