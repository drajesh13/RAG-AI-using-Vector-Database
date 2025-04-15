# #Verify Google Cloud Project and API Key:Correct Project: Double-check that you're using the correct Google Cloud Project where you've enabled the Generative Language API.Valid API Key: Ensure that the GOOGLE_API_KEY you're using is valid and associated with the project that has the API enabled. You can verify your API key in the Google Cloud Console.2. Check Generative Language API Status:API Enabled: Confirm that the Generative Language API is enabled in your Google Cloud Project. You can do this in the Google Cloud Console by searching for "Generative Language API" and making sure it's enabled.3. Model Availability:List Available Models: Although you're specifying "gemini-pro", it's possible that the model name needs to be specified in a different way, or that it's not available in your region. You can use the Google Cloud client library to list the available models and their supported methods.  Here's how you can modify your code to list the models:import streamlit as st
# import streamlit as st
# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY is missing. Please set it in your .env file or as an environment variable.")

# def list_available_models():
#     genai.configure(api_key=GOOGLE_API_KEY)
#     models = genai.list_models()
#     st.write("Available Models:")
#     for m in models:
#         st.write(f"-   Name: {m.name}")
#         st.write(f"    Description: {m.description}")
#         st.write(f"    Supported Generation Methods: {m.supported_generation_methods}")

# # Set Streamlit page
# st.set_page_config(page_title="Check Gemini Models", layout="centered")
# st.title("Available Gemini Models")

# list_available_models()
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import RetrievalQA
from document import load_and_split_documents, generate_embeddings  # Import from your document.py

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing. Please set it in your .env file or as an environment variable.")

# Set Streamlit page
st.set_page_config(page_title="Chat with Your PDF", layout="centered")
st.title("📄 RAGAI — Ask your PDF")

# Initialize LLM
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)  # Keep model="gemini-pro"

# Session state for vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# File upload
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

# Toggle for source documents
show_sources = st.toggle("Show Source Chunks")

# --- INSPECTION POINT 1: After potential loading/creation ---
st.subheader("Vector Store State (Initial/After Upload):")
if st.session_state.vector_store:
    st.info("Vector store is initialized.")
    num_vectors = st.session_state.vector_store._collection.count()
    st.write(f"Number of vectors: {num_vectors}")
else:
    st.warning("Vector store is not yet initialized.")
# --- END INSPECTION POINT 1 ---

# Process PDF and create/load vector store
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        try:
            document_chunks = load_and_split_documents(uploaded_file)
            embeddings_model = generate_embeddings()  # Use your embedding function

            if st.session_state.vector_store is None:
                st.session_state.vector_store = Chroma.from_documents(
                    document_chunks, embeddings_model, persist_directory="chroma_db"
                )
                st.info("Vector store created and populated.")
            else:
                st.session_state.vector_store.add_documents(document_chunks) #add new docs
                st.info("New documents added to the vector store.")
            st.success("PDF processed!")

            # --- INSPECTION POINT 2: After processing PDF ---
            st.subheader("Vector Store State (After Processing PDF):")
            if st.session_state.vector_store:
                num_vectors = st.session_state.vector_store._collection.count()
                st.write(f"Number of vectors: {num_vectors}")
                try:
                    all_data = st.session_state.vector_store._collection.get(include=["documents", "metadatas"], limit=2) # Limit to 2 for brevity
                    if all_data['documents']:
                        st.write("First few documents and metadata:")
                        for i in range(len(all_data['documents'])):
                            st.write(f"Document {i+1}: {all_data['documents'][i][:100]}...") # Show first 100 chars
                            st.write(f"Metadata {i+1}: {all_data['metadatas'][i]}")
                    else:
                        st.info("No documents found in the vector store.")
                except Exception as e:
                    st.error(f"Error retrieving data: {e}")
            else:
                st.warning("Vector store is still not initialized.")
            # --- END INSPECTION POINT 2 ---

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.stop()

# Set up RetrievalQA chain
if st.session_state.vector_store:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vector_store.as_retriever(),
        return_source_documents=show_sources,  # Use the toggle value
    )
else:
        qa_chain = None

# UI Input and Get Answer
user_query = st.text_input("Ask a question about your document:")
if st.button("Get Answer"):
    if user_query.strip():
        if qa_chain:
            response = qa_chain({"query": user_query})
            st.subheader("Answer:")
            st.write(response["result"])

            # Show sources
            if show_sources:
                st.subheader("Source Chunks:")
                for i, doc in enumerate(response["source_documents"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.info(doc.page_content)
        else:
            st.warning("Please upload a PDF to ask a question.")
    else:
        st.warning("Please enter a valid question.")