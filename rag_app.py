import streamlit as st
import tempfile
import os
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Document Q&A", layout="wide")
st.title("📄 Smart Document Q&A System (High Accuracy Version)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


@st.cache_resource
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 🔥 Better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(docs)

    # 🔥 Stronger embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
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

        #  Retrieve more relevant chunks
        docs = vectorstore.similarity_search(question, k=5)

        context = "\n\n".join([doc.page_content for doc in docs])

        #  HuggingFace Router API (Stable)
        API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"

        headers = {
            "Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"
        }

        prompt = f"""
You are a professional document analyst.

Answer the question using ONLY the context below.
If the answer exists in the context, extract it clearly.
If it does not exist, say: Not found in document.

Context:
{context}

Question:
{question}
"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0,
                "max_new_tokens": 512
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            answer = result[0]["generated_text"]
        else:
            answer = f"API Error: {response.status_code} - {response.text}"

        st.subheader("Answer")
        st.write(answer)

