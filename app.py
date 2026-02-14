import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
st.set_page_config(page_title="Local RAG (No Pinecone)", layout="wide")
st.title("🤖 Groq RAG (Local/Offline DB)")

# --- 1. SETUP (Secrets) ---
# You only need the Groq key now. The Database is local.
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")

if not groq_api_key:
    st.info("Please add your Groq API key to continue.")
    st.stop()

client = Groq(api_key=groq_api_key)

# --- 2. INITIALIZE EMBEDDINGS ---
# We use LangChain's wrapper for FastEmbed so it works with FAISS
@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

embeddings = get_embeddings()

# --- 3. SIDEBAR: INGESTION ---
with st.sidebar:
    st.header("Upload Knowledge")
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    
    if uploaded_file and st.button("Ingest Data"):
        with st.spinner("Processing..."):
            # A. Read Text
            text = uploaded_file.read().decode("utf-8")
            
            # B. Chunk Text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.create_documents([text])
            
            # C. Create Local Vector Store (FAISS)
            # This creates the "Database" in your RAM
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            # D. Save to Disk (Optional, so you don't lose it on reload)
            vector_store.save_local("faiss_index")
            
            st.success(f"✅ Indexed {len(chunks)} chunks locally!")

# --- 4. RELOAD VECTOR STORE ---
# Try to load the existing database from disk if it exists
if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = None

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your uploaded data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if vector_store is None:
        st.warning("⚠️ No data found! Please upload a file in the sidebar first.")
        st.stop()

    # RAG Logic
    with st.spinner("Thinking..."):
        # 1. Retrieve (Search Local FAISS)
        # k=3 means "get top 3 most relevant chunks"
        results = vector_store.similarity_search(prompt, k=3)
        context_text = "\n\n".join([doc.page_content for doc in results])
        
        # 2. Generate (Send to Groq)
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided Context to answer the user."},
            {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {prompt}"}
        ]
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="qwen-2.5-32b", 
            temperature=0.5,
        )
        
        response = chat_completion.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
