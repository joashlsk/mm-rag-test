import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Summarizer", layout="centered")
st.title("📄 PDF Assistant")

# --- 1. SETUP ---
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

def get_pdf_text(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text, None
    except Exception as e:
        return None, str(e)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("Upload Document")
    
    # API Key Input (Hidden if using Secrets)
    if "GROQ_API_KEY" not in st.secrets:
        api_key_input = st.text_input("Groq API Key", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input

    # File Uploader
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    
    # Session State Init
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Processing Button
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Reading & Indexing..."):
            text, error = get_pdf_text(uploaded_file)
            
            if error:
                st.error(f"Error: {error}")
            elif not text:
                st.warning("PDF appears empty.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.create_documents([text])
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                st.success("✅ Ready to chat!")

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF and ask me to summarize it!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ex: Summarize this document..."):
    # User Message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Validation
    if st.session_state.vector_store is None:
        st.error("Please upload and process a PDF first.")
        st.stop()
    
    client = get_groq_client()
    if not client:
        st.error("API Key missing.")
        st.stop()

    # RAG Retrieval (Hidden from UI)
    docs = st.session_state.vector_store.similarity_search(query, k=5)
    context_text = "\n\n".join([d.page_content for d in docs])

    # AI Generation
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based ONLY on the context provided below. If asked to summarize, summarize the context."},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant", # Safe, fast model
            temperature=0.3,
            stream=True
        )
        response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
