import streamlit as st
import tempfile
import os
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Document AI", layout="wide")
st.title("📄 Smart Document AI (Multi-PDF + Accurate Version)")

# ---------------- TOKEN DEBUG ----------------
st.write("Token Loaded:", "HUGGINGFACEHUB_API_TOKEN" in os.environ)

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    st.error("❌ HuggingFace token not found in Secrets.")
    st.stop()

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ---------------- DOCUMENT PROCESSING ----------------
@st.cache_resource
def process_documents(files):
    all_docs = []

    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


if uploaded_files:

    vectorstore = process_documents(uploaded_files)
    st.success("✅ Documents processed successfully!")

    mode = st.selectbox(
        "Choose Mode",
        ["Ask Question", "Resume Scoring (1–10)", "ATS Match System"]
    )

    user_input = st.text_area("Enter your question or job description")

    if user_input:

        docs = vectorstore.similarity_search(user_input, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        if mode == "Ask Question":
            prompt = f"""
Answer ONLY from the context below.
If not found, say 'Not found in document'.

Context:
{context}

Question:
{user_input}
"""

        elif mode == "Resume Scoring (1–10)":
            prompt = f"""
You are a professional HR evaluator.

Based on the resume content below,
give a resume score from 1 to 10.

Explain strengths and weaknesses clearly.

Resume Content:
{context}
"""

        else:  # ATS Mode
            prompt = f"""
You are an ATS system.

Compare this resume with the job description.

Give:
- Match percentage
- Missing skills
- Improvement suggestions

Resume:
{context}

Job Description:
{user_input}
"""

        # ---------------- HF ROUTER API ----------------
        API_URL = "https://router.huggingface.co/hf-inference/models/HuggingFaceH4/zephyr-7b-beta"

        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 700,
                "temperature": 0.2,
                "return_full_text": False
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            answer = result[0]["generated_text"]
        else:
            answer = f"API Error: {response.status_code}\n\n{response.text}"

        st.subheader("Result")
        st.write(answer)
