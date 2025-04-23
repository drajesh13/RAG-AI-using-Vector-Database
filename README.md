
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

What is Retrieval-Augmented Generation (RAG)?
RAG allows Large Language Models (LLMs) to generate answers using external data sources that are not part of their original training data.
Why is RAG needed?
 – LLMs are trained on static datasets and cannot access real-time, domain-specific, or private data.
 – Fine-tuning a model for every knowledge update is expensive and inefficient.
 – RAG provides up-to-date and accurate answers without retraining the model.
How does RAG work?
–A user submits a query.
–The query is converted into vector form using the same embedding model used for the document chunks.
–The vector is compared against a Vector Database to retrieve relevant document chunks.
–These chunks are appended to the prompt.
–The LLM generates a response using both the query and retrieved context.
Key Components in a RAG System:
 – Embedding Models: HuggingFace, OpenAI, Gemini Embeddings
 – Vector Databases: FAISS, Chroma, Weaviate, Pinecone
 – LLMs: GPT-4,Gemini, Mistral, LLaMA
 – Frameworks: LangChain, LlamaIndex, Haystack
Benefits of RAG:
 – Provides real-time, context-aware responses
 – Reduces hallucinations
 – Avoids costly fine-tuning
 – Works across a wide range of domains (PDF Q&A, support chatbots, documentation agents)

