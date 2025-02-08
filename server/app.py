from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Chatbot", version="1.0")

# Define request model for the input
class UserRequest(BaseModel):
    topic: str

# Initialize the LLM with Ollama model
llm = Ollama(model="llama2")

# Create the ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("Ask me anything: {topic}")

# Define the route to handle user requests
@app.post("/")
async def get_response(request: UserRequest):
    chain = prompt | llm
    response = chain.invoke({"topic": request.topic})
    return {"output": response}

if __name__ == "__main__":
    # Run FastAPI app with uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)