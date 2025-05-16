import os
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq

st.set_page_config(page_title="LangGraph Chatbot", layout="centered")
st.title("🤖 LangGraph Chatbot")

# Apply combined custom CSS for both UI improvements and darkening the text
st.markdown("""
    <style>
    #Main,header,footer {visibility:hidden;}

    /* General background color for the app */
    .stApp {
        background-color:#edf7f7
        padding: 1rem;
    }

    /* Text and font size improvements */
    html, body, [class*="css"] {
        font-size: 18px !important;
        color: #ffffff !important;  /* White text for contrast */
    }

    /* User and Bot message styling */
    .user-msg {
        background-color: #daf0ff;  /* Light blue background for user messages */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        text-align: left;
    }
    .bot-msg {
        background-color: #e4ffe0;  /* Light green background for bot messages */
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        text-align: left;
    }

    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        .stApp {
            padding: 0.5rem;
        }
        html, body, [class*="css"] {
            font-size: 20px !important;
            color: #000000 !important; /* Darker text for mobile */
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate API key
if not groq_api_key:
    st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
    st.stop()

# Define state
class state(TypedDict):
    message: Annotated[list, add_messages]

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-it")

# Chatbot function
def chatbot(state: state):
    return {'message': llm.invoke(state['message'])}

# Create graph
graph = StateGraph(state)
graph.add_node('chatbot', chatbot)
graph.add_edge(START, 'chatbot')
graph.add_edge('chatbot', END)
my_chatbot = graph.compile()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Input box for user message
user_input = st.chat_input("Type your message here...")

# Process user message
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.spinner("Thinking..."):
        for event in my_chatbot.stream({'message': [user_input]}):
            for value in event.values():
                bot_reply = value['message'].content
                st.session_state.chat_history.append(("bot", bot_reply))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f'<div class="user-msg">👤 You : {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 Bot : {msg}</div>', unsafe_allow_html=True)
