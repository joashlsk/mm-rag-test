import streamlit as st
import os
import time
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fast FAISS RAG", layout="wide")
st.title("⚡ FAISS Neural Search & RAG")

# --- 1. SETUP & CACHING ---
@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def get_groq_client():
    groq_api_key = None
    if "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    elif "GROQ_API_KEY" in os.environ:
        groq_api_key = os.environ["GROQ_API_KEY"]
        
    if not groq_api_key:
        return None
    return Groq(api_key=groq_api_key)

embeddings = get_embeddings()

# --- 2. SIDEBAR: CONFIG & DATA ---
with st.sidebar:
    st.header("⚙️ Configuration")
    if "GROQ_API_KEY" not in st.secrets:
        api_key_input = st.text_input("Enter Groq API Key", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            st.success("Key Set!")
    
    st.divider()
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Upload .txt or .md", type=["txt", "md"])
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if uploaded_file and st.button("Build Fast Index"):
        with st.spinner("🚀 Indexing..."):
            text = uploaded_file.read().decode("utf-8")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.create_documents([text])
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            st.success(f"Index Built! ({len(chunks)} chunks)")

# --- 3. MAIN SEARCH INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask about your document..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a file and click 'Build Fast Index' first.")
        st.stop()

    start_time = time.time()
    docs = st.session_state.vector_store.similarity_search(query, k=4)
    retrieval_time = time.time() - start_time

    with st.expander(f"🔍 Fast Retrieval Debug ({retrieval_time:.4f}s)", expanded=False):
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.caption(doc.page_content[:300] + "...")
            st.divider()

    client = get_groq_client()
    if not client:
        st.error("❌ Groq API Key missing.")
        st.stop()
        
    context_text = "\n\n".join([d.page_content for d in docs])
    
    messages = [
        {"role": "system", "content": "Answer the user question based ONLY on the context below."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                messages=messages,
                # FIX: Switched to Llama 3.3 70B (Current Stable Model)
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                stream=True
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Groq API Error: {e}")
