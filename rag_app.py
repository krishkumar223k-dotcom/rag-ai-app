import streamlit as st
import tempfile
import os
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Smart Document AI", layout="wide")
st.title("📄 Smart Document AI (Multi-PDF + Accurate Version)")

# ---------------- CHECK TOKEN ----------------
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    st.error("❌ HuggingFace token not found in Secrets.")
    st.stop()

HF_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]
st.success("✅ Token Loaded Successfully")

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

@st.cache_resource
def process_documents(files):
    all_docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


if uploaded_files:

    vectorstore = process_documents(uploaded_files)
    st.success("📄 Documents processed successfully!")

    question = st.text_area("Enter your question")

    if question:

        # -------- RETRIEVE CONTEXT --------
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        # -------- HUGGINGFACE ROUTER API --------
        API_URL = "https://router.huggingface.co/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "HuggingFaceH4/zephyr-7b-beta",
            "messages": [
                {
                    "role": "system",
                    "content": "Answer ONLY from the provided context. If not found, say 'Not found in document.'"
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}"
                }
            ],
            "temperature": 0.2,
            "max_tokens": 600
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        # -------- DEBUG RESPONSE --------
        st.write("🔎 Status Code:", response.status_code)

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
        else:
            answer = f"API Error: {response.status_code}\n{response.text}"

        st.subheader("Result")
        st.write(answer)
