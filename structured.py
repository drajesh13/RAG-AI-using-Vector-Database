# import streamlit as st
# import os
# from dotenv import load_dotenv
# from langchain_community.vectorstores import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
# from langchain.chains import RetrievalQA
# from langchain.docstore.document import Document  # Import Document
# from document import generate_embeddings  # Assuming you have this
# import pandas as pd

# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     st.error("GOOGLE_API_KEY is missing.")

# genai.configure(api_key=GOOGLE_API_KEY)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
# embeddings_model = generate_embeddings()

# persist_directory = "chroma_db_structured"  # Separate directory for structured data

# if "vector_store_structured" not in st.session_state:
#     st.session_state.vector_store_structured = None

# uploaded_file_structured = st.file_uploader("Upload your structured data (e.g., CSV)", type=["csv"])

# if uploaded_file_structured is not None:
#     with st.spinner("Processing structured data..."):
#         try:
#             df = pd.read_csv(uploaded_file_structured)
#             total_records = len(df)
#             documents = []
#             for index, row in df.iterrows():
#                 text_representation = ", ".join([f"{col}: {row[col]}" for col in df.columns])
#                 metadata = {"row_index": index}
#                 documents.append(Document(page_content=text_representation, metadata=metadata))

#             # Add a summary document with the total record count
#             summary_document = Document(page_content=f"This dataset contains a total of {total_records} records.", metadata={"type": "summary", "total_records": total_records})
#             documents.append(summary_document)

#             if st.session_state.vector_store_structured is None:
#                 st.session_state.vector_store_structured = Chroma.from_documents(
#                     documents, embeddings_model, persist_directory=persist_directory
#                 )
#             else:
#                 st.session_state.vector_store_structured.add_documents(documents)
#             st.success("Structured data processed!")

#         except Exception as e:
#             st.error(f"Error processing structured data: {e}")

# # Querying the structured data
# structured_query = st.text_input("Ask about your structured data:")
# if st.button("Get Structured Data Answer"):
#     if structured_query.strip() and st.session_state.vector_store_structured:
#         retriever = st.session_state.vector_store_structured.as_retriever(
#             search_kwargs={'k': 5}  # Adjust k as needed
#         )
#         relevant_documents = retriever.get_relevant_documents(structured_query)

#         # Check if the query is specifically about the total number of records
#         if "total number of records" in structured_query.lower() or "how many records" in structured_query.lower():
#             for doc in relevant_documents:
#                 if doc.metadata.get("type") == "summary" and "total_records" in doc.metadata:
#                     st.subheader("Total Number of Records:")
#                     st.info(f"{doc.metadata['total_records']}")
#                     break  # Assuming only one summary document
#             else:
#                 # Fallback to LLM if summary isn't found (shouldn't happen if implemented correctly)
#                 qa_chain_structured = RetrievalQA.from_chain_type(
#                     llm=llm,
#                     chain_type="stuff",
#                     retriever=retriever,
#                     return_source_documents=True
#                 )
#                 response_structured = qa_chain_structured({"query": structured_query})
#                 st.subheader("Structured Data Answer:")
#                 st.write(response_structured["result"])
#                 if st.toggle("Show Source Rows"):
#                     st.subheader("Source Rows:")
#                     for doc in response_structured["source_documents"]:
#                         st.info(f"Row: {doc.page_content}")
#                         st.json(doc.metadata)
#         else:
#             # For other queries, use the regular RetrievalQA
#             qa_chain_structured = RetrievalQA.from_chain_type(
#                 llm=llm,
#                 chain_type="stuff",
#                 retriever=retriever,
#                 return_source_documents=True
#             )
#             response_structured = qa_chain_structured({"query": structured_query})
#             st.subheader("Structured Data Answer:")
#             st.write(response_structured["result"])
#             if st.toggle("Show Source Rows"):
#                 st.subheader("Source Rows:")
#                 for doc in response_structured["source_documents"]:
#                     st.info(f"Row: {doc.page_content}")
#                     st.json(doc.metadata)

#     elif not uploaded_file_structured:
#         st.warning("Please upload structured data first.")
#     else:
#         st.warning("Please enter a valid question.")

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document  # Import Document
from document import generate_embeddings  # Assuming you have this
import pandas as pd
import re  # For simple pattern matching
from io import StringIO

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing.")

genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
embeddings_model = generate_embeddings()

persist_directory = "chroma_db_structured_meta_v4"  # Separate directory

if "vector_store_structured" not in st.session_state:
    st.session_state.vector_store_structured = None

uploaded_file_structured = st.file_uploader("Upload your structured data (e.g., CSV)", type=["csv"])
has_header = st.checkbox("My CSV file has a header row", value=True)
csv_delimiter = st.text_input("CSV Delimiter (e.g., ',', ';', '\\t'):", ",")
csv_encoding = st.text_input("CSV Encoding (e.g., 'utf-8', 'latin-1'):", "utf-8")

if uploaded_file_structured is not None:
    with st.spinner("Processing structured data..."):
        try:
            header = 0 if has_header else None
            stringio = StringIO(uploaded_file_structured.getvalue().decode(csv_encoding))
            df = pd.read_csv(stringio, header=header, sep=csv_delimiter)

            if df.empty:
                st.error("The uploaded CSV file is empty.")
                st.stop()

            if header is None:
                df.columns = [f"col_{i}" for i in range(len(df.columns))] # Assign default column names

            st.session_state["df_columns"] = df.columns.tolist()
            documents = []
            for index, row in df.iterrows():
                text_representation = ", ".join([f"{col}: {row[col]}" for col in df.columns])
                metadata = row.to_dict()  # Store all column values as metadata
                documents.append(Document(page_content=text_representation, metadata=metadata))

            collection_metadata = {"total_records": len(df)}

            if st.session_state.vector_store_structured is None:
                st.session_state.vector_store_structured = Chroma.from_documents(
                    documents, embeddings_model, persist_directory=persist_directory, collection_metadata=collection_metadata
                )
            else:
                st.session_state.vector_store_structured.add_documents(documents)
            st.success("Structured data processed with detailed metadata!")

        except Exception as e:
            st.error(f"Error processing structured data: {e}")

def handle_structured_query(vector_store, query, df_columns):
    if not vector_store:
        return False, None

    if "total number of records" in query.lower() or "how many rows" in query.lower():
        collection = vector_store._collection
        metadata = collection.metadata
        if metadata and "total_records" in metadata:
            st.subheader("Total Number of Records:")
            st.info(f"{metadata['total_records']}")
            return True, None
        else:
            st.warning("Total record count metadata not found.")
            return True, None  # Handled, but no count found

    # Attempt simple equality-based metadata filtering
    for col in df_columns:
        match = re.search(rf"how many .* where {col.lower()} is (.*)", query.lower())
        if match:
            value = match.group(1).strip()
            try:
                results = vector_store.get(where={col: value}, include=[])
                st.subheader(f"Number of records where {col} is '{value}':")
                st.info(f"{len(results['ids'])}")
                return True, None
            except Exception as e:
                st.warning(f"Error during metadata filtering for '{col}': {e}")
                break # Try other patterns

    # If no specific structured query handled, fall back to RAG
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    response_structured = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    response = response_structured({"query": query})
    st.subheader("Structured Data Answer:")
    st.write(response["result"])
    if st.toggle("Show Source Rows"):
        st.subheader("Source Rows:")
        for doc in response["source_documents"]:
            st.info(f"Row: {doc.page_content}")
            st.json(doc.metadata)
    return True, response # Indicate RAG was used

# Querying the structured data
structured_query = st.text_input("Ask about your structured data:")
if st.button("Get Structured Data Answer"):
    if structured_query.strip() and st.session_state.vector_store_structured and st.session_state.get("df_columns"):
        success, _ = handle_structured_query(st.session_state.vector_store_structured, structured_query, st.session_state["df_columns"])
    elif not uploaded_file_structured:
        st.warning("Please upload structured data first.")
    else:
        st.warning("Please enter a valid question.")

# Store column names in session state for easier access
if uploaded_file_structured and "df_columns" not in st.session_state:
    try:
        stringio = StringIO(uploaded_file_structured.getvalue().decode("utf-8"))
        # Read a sample to get columns without fully processing
        temp_df = pd.read_csv(stringio, nrows=1, sep=csv_delimiter, header=0 if has_header else None)
        st.session_state["df_columns"] = temp_df.columns.tolist()
    except Exception as e:
        st.error(f"Error reading column names: {e}")