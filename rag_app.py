import streamlit as st
import tempfile
import os
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.title("🚀 AI Resume Analyzer + Multi PDF Chat")

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

mode = st.selectbox(
    "Choose Mode",
    ["Document Q&A", "Resume Scoring (1–10)", "ATS Match System"]
)

# ---------------- PROCESS DOCUMENTS ----------------
@st.cache_resource
def process_documents(files):
    all_docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


# ---------------- HF ROUTER CONFIG ----------------
API_URL = "https://router.huggingface.co/hf-inference/models/HuggingFaceH4/zephyr-7b-beta"

headers = {
    "Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"
}

# ---------------- MAIN ----------------
if uploaded_files:

    vectorstore = process_documents(uploaded_files)
    st.success("Documents processed successfully!")

    user_input = st.text_area("Enter your question or job description")

    if user_input:

        docs = vectorstore.similarity_search(user_input, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Mode-based prompt
        if mode == "Document Q&A":
            prompt = f"""
Answer using ONLY the context below.
If answer not found, say 'Not found in document'.

Context:
{context}

Question:
{user_input}
"""

        elif mode == "Resume Scoring (1–10)":
            prompt = f"""
You are a resume evaluator.

Based on the resume below, give:
1. Resume score from 1 to 10
2. Strengths
3. Weaknesses
4. Improvement suggestions

Resume:
{context}
"""

        else:  # ATS Match
            prompt = f"""
You are an ATS system.

Compare the resume with the job description.

Resume:
{context}

Job Description:
{user_input}

Give:
1. ATS Match Percentage
2. Missing Keywords
3. Matching Skills
4. Suggestions to improve match
"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 600,
                "temperature": 0.3,
                "return_full_text": False
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            answer = result[0]["generated_text"]
        else:
            answer = f"API Error: {response.status_code} - {response.text}"

        st.subheader("Result")
        st.write(answer)