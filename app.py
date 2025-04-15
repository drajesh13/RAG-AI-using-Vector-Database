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
            else:
                st.session_state.vector_store.add_documents(document_chunks) #add new docs
            st.success("PDF processed!")
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


# import streamlit as st
# from langchain.chains import RetrievalQA
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma  # Changed import
# from document import load_and_split_documents, generate_embeddings, load_vector_store # Import from document.py

# # --- Load environment variables ---
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY is missing. Please set it in your .env file or as an environment variable.")

# # --- Page configuration ---
# st.set_page_config(page_title="Chat with Your PDF", page_icon=":books:")

# # --- Session state initialization ---
# if "vector_store" not in st.session_state:
#     st.session_state.vector_store = None
# if "qa_chain" not in st.session_state:
#     st.session_state.qa_chain = None
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # --- Helper functions ---
# def initialize_llm():
#     """Initializes the Gemini Pro LLM."""
#     genai.configure(api_key=GOOGLE_API_KEY)
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)  # Changed model_name to model
#     return llm

# def create_retrieval_chain(llm, vector_store):
#     """Creates the retrieval chain."""
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Adjust k as needed
#         return_source_documents=True, # Return source documents for better context
#     )
#     return qa_chain

# def ask_question(chain, query):
#     """Asks a question to the retrieval chain."""
#     result = chain({"query": query})
#     return result

# def display_chat_history(chat_history):
#     """Displays the chat history in the Streamlit app."""
#     for role, message in chat_history:
#         with st.chat_message(role):
#             st.write(message)

# # --- Main App ---
# def main():
#     st.title("Chat with Your PDF using Gemini and Chroma")

#     # --- File Upload Section ---
#     uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

#     # --- Process PDF and Initialize Vector Store ---
#     if uploaded_file is not None:
#         with st.spinner("Processing PDF..."):
#             try:
#                 # 1. Load and split documents
#                 document_chunks = load_and_split_documents(uploaded_file)

#                 # 2. Generate embeddings
#                 embeddings_model = generate_embeddings()

#                 # 3. Create or load vector store
#                 if st.session_state.vector_store is None:
#                     st.session_state.vector_store = Chroma.from_documents(document_chunks, embeddings_model, persist_directory="chroma_db")
#                     # st.session_state.vector_store.persist()  # Removed persist()
#                 else:
#                     st.session_state.vector_store = load_vector_store(embeddings_model)

#                 st.success("PDF processed and embeddings generated!")
#             except Exception as e:
#                 st.error(f"Error processing PDF: {e}")
#                 return  # Stop processing if there's an error

#         # --- Initialize LLM and Chain ---
#         if st.session_state.vector_store is not None:
#             llm = initialize_llm()
#             if st.session_state.qa_chain is None:
#                 st.session_state.qa_chain = create_retrieval_chain(llm, st.session_state.vector_store)

#         # --- Chat Input and Interaction ---
#         query = st.chat_input("Ask a question about your PDF:")
#         if query:
#             if st.session_state.qa_chain is not None:
#                 try:
#                     with st.spinner("Generating response..."):
#                         result = ask_question(st.session_state.qa_chain, {"query": query})

#                         # --- Debugging Output ---
#                         st.write(f"Type of result: {type(result)}")
#                         st.write(f"Keys in result: {result.keys()}")
#                         if "result" in result:
#                             st.write(f"Type of result['result']: {type(result['result'])}")
#                             st.write(f"Value of result['result']: {result['result']}")
#                         if "source_documents" in result:
#                             st.write(f"Type of result['source_documents']: {type(result['source_documents'])}")
#                             st.write(f"Length of result['source_documents']: {len(result['source_documents'])}")
#                         if result["source_documents"]:
#                             st.write(f"Type of first source_document: {type(result['source_documents'][0])}")
#                             if hasattr(result["source_documents"][0], 'page_content'):
#                                 st.write(f"Type of page_content: {type(result['source_documents'][0].page_content)}")
#                                 st.write(f"First 100 chars of page_content: {result['source_documents'][0].page_content[:100]}")
#                             else:
#                                 st.write("First source_document has no page_content attribute.")
#                         else:
#                             st.write("Key 'source_documents' not in result.")
#                         # --- End Debugging Output ---

#                         answer = result.get("result", "No answer found.")
#                         st.session_state.chat_history.append(("user", query))
#                         st.session_state.chat_history.append(("assistant", answer))

#                         st.subheader("Answer:")
#                         st.write(answer)

#                         # Display source documents
#                         with st.expander("Show Source Documents"):
#                             for doc in result.get("source_documents", []):
#                                 st.write(doc.page_content)
#                                 st.write("---")

#                     display_chat_history(st.session_state.chat_history)

#                 except Exception as e:
#                     st.error(f"Error generating response: {e}")
#             else:
#                 st.warning("Please upload a PDF to ask questions.")
#         else:
#             display_chat_history(st.session_state.chat_history)

# if __name__ == "__main__":
#     main()

