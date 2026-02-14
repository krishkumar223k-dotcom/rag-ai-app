
import streamlit as st
import tempfile
import os
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Smart Document Q&A", layout="wide")
st.title("📄 Smart Document Q&A System (Cloud Version)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


@st.cache_resource
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    vectorstore = process_document(tmp_path)
    st.success("Document processed successfully!")

    question = st.text_input("Ask a question about your document")

    if question:

        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        final_prompt = f"""
Answer the question using ONLY the context below.
If the answer is not found, say 'Not found in document'.

Context:
{context}

Question:
{question}
"""

        API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"

        headers = {
            "Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": final_prompt,
            "parameters": {
                "temperature": 0,
                "max_new_tokens": 512
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            st.error(f"API Error: {response.text}")
        else:
            result = response.json()

            if isinstance(result, list) and "generated_text" in result[0]:
                answer = result[0]["generated_text"]
                st.subheader("Answer")
                st.write(answer)
            else:
                st.error(f"Unexpected API Response: {result}")
