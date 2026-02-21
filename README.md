# mm-rag-test
streamlit (Front end UI)
FAISS (Facebook AI Similarity Search)


app.py/                     # Your Monolithic Streamlit Application
│
├── 0. Imports/             # External library dependencies
│   ├── streamlit           # UI framework
│   ├── groq                # LLM API client
│   ├── langchain_* # Vector store, embeddings, text splitters
│   └── pypdf               # PDF text extraction
│
├── 1. Page Config/         # UI initialization
│   └── st.set_page_config  # Sets page title and wide layout
│
├── 2. Setup & Caching/     # Backend initialization functions
│   ├── get_embeddings()    # Caches the BAAI FastEmbed model
│   ├── get_groq_client()   # Retrieves and validates the API key
│   └── get_pdf_text()      # Safely extracts text from uploaded PDFs
│
├── 3. Sidebar Config/      # Left-hand UI panel & Data Ingestion
│   ├── API Key Input       # Fallback input for missing Groq keys
│   ├── Model Selection     # Dropdown for Llama/Mixtral models
│   ├── File Uploader       # Accepts .txt and .pdf files
│   └── Build Index Logic   # Triggers document reading, chunking, and FAISS vector storage
│
└── 4. Main Interface/      # Chat UI & RAG Execution
    ├── Chat History        # Loops through and renders past st.session_state messages
    ├── User Input          # Captures the user's question via st.chat_input
    ├── Retrieval Step      # FAISS similarity search & renders the debugging expander
    └── Generation Step     # Assembles the system prompt, calls Groq, and streams the response
