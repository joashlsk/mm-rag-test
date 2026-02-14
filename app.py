import streamlit as st
import os
import time
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIG ---
st.set_page_config(page_title="RAG Debugger", layout="wide")
st.title("🛠️ RAG System Diagnostic Mode")

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
    
    # Model Selector (In case one is down)
    model_option = st.selectbox(
        "Select Model:",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )
    
    st.divider()
    st.header("📂 1. Upload Document")
    uploaded_file = st.file_uploader("Upload .txt or .md", type=["txt", "md"])
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if uploaded_file and st.button("Build Index"):
        with st.spinner("🚀 Indexing..."):
            try:
                text = uploaded_file.read().decode("utf-8")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.create_documents([text])
                
                # Force new index creation
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.success(f"✅ Index Built! Processed {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# --- 3. MAIN INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question (or type 'summarize')..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vector_store is None:
        st.error("⚠️ No Index Found. Please upload a file and click 'Build Index' in the sidebar.")
        st.stop()

    # --- RETRIEVAL STEP ---
    # k=5 for more context
    docs = st.session_state.vector_store.similarity_search(query, k=5)
    
    # DEBUG: Show what was found
    with st.expander("🔍 Debug: What did the RAG find?", expanded=True):
        if not docs:
            st.warning("❌ No relevant text found in document.")
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:150]}...")

    # --- GENERATION STEP ---
    client = get_groq_client()
    if not client:
        st.error("❌ Groq API Key missing.")
        st.stop()
        
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # Strict System Prompt
    system_prompt = f"""
    You are an expert summarizer and assistant.
    You must answer the user's question using ONLY the context provided below.
    If the context does not contain the answer, say "I cannot find that information in the document."
    
    Context:
    {context_text}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                messages=messages,
                model=model_option, # Uses the dropdown selection
                temperature=0.3,
                max_tokens=1024,
                stream=True
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"❌ API Error: {e}")
