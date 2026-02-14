import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader # <--- NEW IMPORT

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

# --- HELPER: SAFE PDF EXTRACTION ---
def get_pdf_text(uploaded_file):
    """
    Reads a PDF file safely.
    Returns: (text, error_message)
    """
    try:
        # pypdf reads directly from the stream (no need to save to disk)
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""  # Handle None if page is empty
        
        # Check if text was actually found (Handle Scanned PDFs)
        if len(text.strip()) == 0:
            return None, "PDF seems empty or scanned (image-only). This tool requires readable text."
            
        return text, None

    except Exception as e:
        return None, f"Failed to read PDF: {str(e)}"

# --- 2. SIDEBAR: CONFIG & DATA ---
with st.sidebar:
    st.header("⚙️ Configuration")
    if "GROQ_API_KEY" not in st.secrets:
        api_key_input = st.text_input("Enter Groq API Key", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            st.success("Key Set!")
    
    model_option = st.selectbox(
        "Select Model:",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0
    )
    
    st.divider()
    st.header("📂 1. Upload Document")
    
    # UPDATE: Accept both TXT and PDF
    uploaded_file = st.file_uploader("Upload File", type=["txt", "pdf"]) #
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if uploaded_file and st.button("Build Index"):
        with st.spinner("🚀 Indexing..."):
            file_text = ""
            error = None

            # LOGIC: Switch based on file type
            if uploaded_file.name.endswith(".pdf"):
                file_text, error = get_pdf_text(uploaded_file)
            else:
                # Fallback for TXT files
                file_text = uploaded_file.read().decode("utf-8")
            
            # Error Handling (Stops the crash)
            if error:
                st.error(f"❌ {error}")
            elif not file_text:
                st.warning("⚠️ File is empty.")
            else:
                # Success path
                try:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = text_splitter.create_documents([file_text])
                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                    st.success(f"✅ Index Built! Processed {len(chunks)} chunks.")
                except Exception as e:
                    st.error(f"Error during chunking: {e}")

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
    docs = st.session_state.vector_store.similarity_search(query, k=5)
    
    with st.expander("🔍 Debug: What did the RAG find?", expanded=True):
        if not docs:
            st.warning("❌ No relevant text found.")
        for i, doc in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:150]}...")

    # --- GENERATION STEP ---
    client = get_groq_client()
    if not client:
        st.error("❌ Groq API Key missing.")
        st.stop()
        
    context_text = "\n\n".join([d.page_content for d in docs])
    
    system_prompt = f"""
    You are an expert summarizer. Answer using ONLY the context below.
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
                model=model_option,
                temperature=0.3,
                max_tokens=1024,
                stream=True
            )
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"❌ API Error: {e}")
