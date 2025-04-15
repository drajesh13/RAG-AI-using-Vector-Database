import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from document import load_vector_store, generate_embeddings  # Import from document.py

# --- Load environment variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Configuration ---
LLM_MODEL_NAME = "gemini-pro"

def create_retrieval_chain(vector_store, llm, return_source_documents=False):
    """Creates a retrieval-based question answering chain."""
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever, return_source_documents=return_source_documents)
    return qa_chain

def ask_question(chain, query):
    """Asks a question to the retrieval chain."""
    result = chain({"query": query})
    return result

if __name__ == "__main__":
    # Example usage:
    print("Loading embeddings...")
    embeddings = generate_embeddings()

    print("Loading vector store...")
    vector_store = load_vector_store(embeddings)

    print("Initializing Gemini Pro model...")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable not set. Please create a .env file or set it directly.")
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY)

    print("Creating retrieval chain...")
    qa_chain = create_retrieval_chain(vector_store, llm, return_source_documents=True)

    while True:
        query = input("Ask your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        print("Processing your question...")
        result = ask_question(qa_chain, query)

        print("\nAnswer:")
        print(result["result"])

        if "source_documents" in result and result["source_documents"]:
            print("\nSource Documents:")
            for i, doc in enumerate(result["source_documents"]):
                print(f"Source {i+1}: {doc.page_content[:100]}...") # Display first 100 chars
                print(f"Metadata: {doc.metadata}")
                print("---")
        print("-" * 30)