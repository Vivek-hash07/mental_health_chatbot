import streamlit as st
import requests

# Function to send the request to FastAPI backend
def get_response(input_text):
    try:
        # Send the request to FastAPI
        response = requests.post("http://127.0.0.1:8000", json={"topic": input_text})
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()["output"]
        else:
            return "Error: Failed to get a response from the server."
    except requests.exceptions.ConnectionError as e:
        return f"Error: Unable to connect to the server. Please ensure the FastAPI server is running. ({e})"

# Streamlit UI
st.title("Alie: The AI Chatbot")
input_text = st.text_input("Enter your question:")

if input_text:
    st.write(get_response(input_text))