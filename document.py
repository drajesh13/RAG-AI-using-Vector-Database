import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile

# --- Load environment variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "models/embedding-001"
CHROMA_DB_PATH = "chroma_db"  # Local directory to store ChromaDB

def load_and_split_documents(file_source):
    """Loads a PDF from a file path or a Streamlit UploadedFile, splits it into chunks."""
    try:
        if isinstance(file_source, str):
            # If a file path is provided (when run directly)
            loader = PyPDFLoader(file_source)
            documents = loader.load()
        else:
            # If a Streamlit UploadedFile object is provided (from app.py)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(file_source.read())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            os.remove(tmp_file_path)  # Clean up temporary file

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_documents(documents)
        return chunks
    except Exception as e:
        raise Exception(f"Error loading and splitting PDF: {e}")

def generate_embeddings():
    """Initializes the GoogleGenerativeAIEmbeddings model."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please create a .env file or set it directly.")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)
    return embeddings

def create_vector_store(chunks, embeddings):
    """Generates embeddings for chunks and stores them in ChromaDB."""
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    vector_store.persist()
    return vector_store

def load_vector_store(embeddings):
    """Loads the Chroma vector store from disk."""
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return vector_store

if __name__ == "__main__":
    # Example usage:
    pdf_file = "/Users/tbommireddy/Desktop/RAG_chat/documents/mhccrules.pdf"  # Use the correct relative path
    if os.path.exists(pdf_file):
        print(f"Loading and splitting document: {pdf_file}")
        document_chunks = load_and_split_documents(pdf_file)
        print(f"Number of chunks: {len(document_chunks)}")

        print("Generating embeddings...")
        embeddings_model = generate_embeddings()

        print(f"Creating vector store in: {CHROMA_DB_PATH}")
        vector_store = create_vector_store(document_chunks, embeddings_model)
        print("Vector store created successfully.")
    else:
        print(f"Error: PDF file not found at {pdf_file}")

