import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = "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"

# Set environment variables (Optional if you are using OpenAI or any other key-based embeddings)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Streamlit framework
st.title("LIEXA L1")

input_text = st.text_input("Enter your question")
data_source = st.radio(
    "Choose Data Source(s)",
    ("PDF", "JSON", "Webpage")
)
#upload the data in form of pdf,jso or web page url
uploaded_file = st.file_uploader("Upload a file (PDF/JSON) or Enter URL for Webpage", type=["pdf", "json", "txt"])

url_input = None
if data_source == "Webpage":
    url_input = st.text_input("Enter Webpage URL")   # enter the web url here

# Load documents based on data source
documents = []

if uploaded_file:
    if data_source == "PDF":
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()

    elif data_source == "JSON":
        loader = JSONLoader(uploaded_file)
        documents = loader.load()

elif url_input and data_source == "Webpage":
    loader = WebBaseLoader(url_input)
    documents = loader.load()

# Split documents into chunks for better retrieval
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

docs = text_splitter.split_documents(documents)

# Create FAISS vector store
embeddings = OpenAIEmbeddings()  # You can swap this with HuggingFaceEmbeddings or any other
vectorstore = FAISS.from_documents(docs, embeddings)

# RAG retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

# Creating prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use the provided context to answer user queries."),
        ("user", "Context: {context}\n\nQuestion: {question}")
    ]
)

# Retrieval + Generation chain
if input_text and docs:
    relevant_docs = retriever.invoke(input_text)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    chain = prompt | llm | output_parser

    response = chain.invoke({"context": context, "question": input_text})
    st.write(f"L1: {response}")

elif input_text and not docs:
    st.warning("Please upload a valid data source first.")
