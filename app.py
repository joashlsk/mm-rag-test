import streamlit as st
import os
import time
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fast FAISS RAG", layout="wide")
st.title("⚡ FAISS Neural Search & RAG")

# --- 1. SETUP & CACHING (The "Fast" Part) ---
# We use st.cache_resource so we only load the heavy models ONCE
@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource
def get_groq_client():
    # Try getting key from secrets, otherwise from user input
    try:
        key = st.secrets["GROQ_API_KEY"]
    except:
        return None
    return Groq(api_key=key)

embeddings = get_embeddings()

# --- 2. SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    
    # Manual Key Entry if not in secrets
    if "GROQ_API_KEY" not in st.secrets:
        api_key_input = st.text_input("Groq API Key", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
    
    uploaded_file = st.file_uploader("Upload .txt or .md", type=["txt", "md"])
    
    # Initialize Session State for the Vector Store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if uploaded_file and st.button("Build Fast Index"):
        with st.spinner("🚀 Indexing..."):
            # A. Read & Split
            text = uploaded_file.read().decode("utf-8")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.create_documents([text])
            
            # B. Build FAISS Index (In-Memory)
            # This is extremely fast because it runs locally on CPU
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            st.success(f"Index Built! ({len(chunks)} chunks)")

# --- 3. MAIN SEARCH INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Query
if query := st.chat_input("Ask about your document..."):
    # 1. Add User Query to Chat
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a file and click 'Build Fast Index' first.")
        st.stop()

    # --- THE "FAST SEARCH" LAYER ---
    start_time = time.time()
    
    # 1. FAISS Retrieval (Milliseconds)
    # k=4 means "Get top 4 most similar chunks"
    docs = st.session_state.vector_store.similarity_search(query, k=4)
    retrieval_time = time.time() - start_time

    # 2. Show Retrieved Context (Neural Search)
    # This allows you to verify what the AI found *before* it answers
    with st.expander(f"🔍 Fast Retrieval Debug ({retrieval_time:.4f}s)", expanded=False):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.caption(doc.page_content[:300] + "...") # Preview first 300 chars
            st.divider()

    # --- THE "GENERATION" LAYER ---
    client = get_groq_client()
    if not client:
        st.error("Groq API Key missing.")
        st.stop()
        
    context_text = "\n\n".join([d.page_content for d in docs])
    
    messages = [
        {"role": "system", "content": "Answer the user question based ONLY on the context below."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            messages=messages,
            model="qwen-2.5-32b",
            temperature=0.3,
            stream=True
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
