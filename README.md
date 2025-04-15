
# 📄 RAG-AI PDF Assistant using Vector Database

This project is a **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, **Google Gemini**, and **ChromaDB**. 

It allows users to upload a PDF, generate embeddings, store them in a vector database, and ask natural language questions — with real-time answers grounded in your document.

🚀 Features

- Upload any PDF and process it with LangChain.
- Automatically split the document into chunks and generate vector embeddings.
- Store embeddings locally using **ChromaDB**.
- Query the document using **Google Gemini (Pro)** via LangChain's `RetrievalQA`.
- Toggle source document visibility for transparent answers.

🧠 How It Works:

Ingest (documents.py):

Loads PDF content using PyPDFLoader from LangChain.
Splits the content into overlapping chunks using RecursiveCharacterTextSplitter (default: 1000 tokens, 100 overlap).
Generates vector embeddings using Google Generative AI Embeddings (embedding-001 model via langchain_google_genai).
Stores vectors in a local ChromaDB (persisted on disk in the chroma_db/ directory).

Query (query.py or app.py):

Accepts a user question via 
Embeds the query using the same embedding model.
Retrieves the top matching chunks from Chroma using vector similarity search.
Sends the retrieved context + query to an LLM
Displays the generated answer and optionally shows the source chunks that supported the response.

## Tech Stack

- Frontend: Streamlit
- LLM: Gemini Pro via `langchain_google_genai`
- Embeddings: GoogleGenerativeAIEmbeddings
- Vector DB: Chroma
- PDF Processing: LangChain's `PyPDFLoader’
- Python
