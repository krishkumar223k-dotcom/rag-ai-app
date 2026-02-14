import streamlit as st
import tempfile
import os
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="AI Resume Analyzer + Multi PDF Chat", layout="wide")
st.title("🚀 AI Resume Analyzer & Multi-Document Chat")

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- DOCUMENT PROCESSING ----------------
@st.cache_resource
def process_documents(file_paths):
    all_docs = []

    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


if uploaded_files:

    temp_paths = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_paths.append(tmp_file.name)

    vectorstore = process_documents(temp_paths)
    st.success("Documents processed successfully!")

    mode = st.selectbox(
        "Choose Mode",
        ["Chat with Documents", "Resume Scoring (1–10)", "ATS Match System"]
    )

    user_input = st.text_area("Enter your question or job description")

    if user_input:

        docs = vectorstore.similarity_search(user_input, k=6)
        context = "\n\n".join([doc.page_content for doc in docs])

        # ---------------- MODE PROMPTS ----------------
        if mode == "Chat with Documents":
            prompt = f"""
You are an AI document assistant.

Answer strictly using the context.
If not found, say 'Not found in document.'

Context:
{context}

Question:
{user_input}
Answer:
"""

        elif mode == "Resume Scoring (1–10)":
            prompt = f"""
You are a professional HR evaluator.

Based on the resume below, give:

1. Resume Score (1–10)
2. Strengths
3. Weaknesses
4. Suggestions for improvement

Resume:
{context}
"""

        elif mode == "ATS Match System":
            prompt = f"""
You are an ATS system.

Compare the resume with the job description.

Return:
1. Match Percentage (0–100%)
2. Missing Keywords
3. Matching Skills
4. Improvement Suggestions

Resume:
{context}

Job Description:
{user_input}
"""

        # ---------------- HF ROUTER API ----------------
        API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"

        headers = {
            "Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 600,
                "temperature": 0.2,
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