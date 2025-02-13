import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

# Set environment variables
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Creating chatbot prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please provide responses to user queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit framework
st.title("LIEXA L1")
input_text = st.text_input("Enter your question")

# LLM call
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write(f"L1: {response}")  # Prepend "L1: " to the response
