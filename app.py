import streamlit as st
from pinecone import Pinecone
from groq import Groq
from fastembed import TextEmbedding
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Enterprise RAG Agent", layout="wide")
st.title("🤖 Enterprise Knowledge Agent")

# --- SIDEBAR: CREDENTIALS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # We use st.secrets for cloud deployment, but allow manual entry for testing
    groq_key = st.text_input("Groq API Key", type="password")
    pinecone_key = st.text_input("Pinecone API Key", type="password")

# --- MAIN LOGIC ---
if not groq_key or not pinecone_key:
    st.warning("⚠️ Please enter your API keys in the sidebar to start.")
    st.stop()

# Initialize Clients
try:
    groq_client = Groq(api_key=groq_key)
    pc = Pinecone(api_key=pinecone_key)
    # Connect to index (Create this in Pinecone dashboard first!)
    index_name = "rag-index"
    if index_name not in pc.list_indexes().names():
        st.error(f"Index '{index_name}' not found in Pinecone.")
    else:
        index = pc.Index(index_name)
        st.success("✅ Connected to Database")
except Exception as e:
    st.error(f"Connection Error: {e}")

# (Your RAG logic goes here - kept short for this file example)
query = st.chat_input("Ask a question about your documents...")
if query:
    st.chat_message("user").write(query)
    # Placeholder response for demo
    st.chat_message("assistant").write(f"I received your query: '{query}'. (RAG logic would run here)")
