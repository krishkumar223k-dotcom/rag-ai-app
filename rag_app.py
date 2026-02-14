import streamlit as st
import tempfile
import requests
import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Resume + RAG Pro", layout="wide")
st.title(" Smart Resume Analyzer + ATS Matcher")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload Resume or PDFs",
    type="pdf",
    accept_multiple_files=True
)

job_description = st.text_area("Optional: Paste Job Description for ATS Matching")

@st.cache_resource
def build_vectorstore(files):
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
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

    return FAISS.from_documents(split_docs, embeddings)

# ---------------- SCORING FUNCTION ----------------
def score_resume(text):
    score = 0

    if len(text) > 1500:
        score += 2

    if re.search(r"experience", text, re.I):
        score += 2

    if re.search(r"skills", text, re.I):
        score += 2

    if re.search(r"education", text, re.I):
        score += 2

    if re.search(r"\d\.\d{1,2}", text):  # CGPA detection
        score += 2

    return min(score, 10)

# ---------------- ATS MATCH FUNCTION ----------------
def ats_match(resume_text, jd_text):
    resume_words = set(re.findall(r"\b\w+\b", resume_text.lower()))
    jd_words = set(re.findall(r"\b\w+\b", jd_text.lower()))

    common = resume_words.intersection(jd_words)
    match_percentage = int((len(common) / max(len(jd_words), 1)) * 100)

    missing = jd_words - resume_words
    missing_keywords = list(missing)[:20]

    return match_percentage, missing_keywords

# ---------------- MAIN ----------------
if uploaded_files:

    vectorstore = build_vectorstore(uploaded_files)
    st.success("Documents processed successfully!")

    # Get full resume text
    docs = vectorstore.similarity_search("resume full content", k=10)
    full_text = "\n".join([doc.page_content for doc in docs])

    # -------- Resume Score --------
    resume_score = score_resume(full_text)
    st.subheader(" Resume Score (1–10)")
    st.progress(resume_score * 10)
    st.write(f"Resume Score: **{resume_score}/10**")

    # -------- ATS Match --------
    if job_description:
        match_percentage, missing_keywords = ats_match(full_text, job_description)

        st.subheader(" ATS Match Percentage")
        st.write(f"Match: **{match_percentage}%**")

        st.subheader(" Missing Keywords")
        st.write(", ".join(missing_keywords) if missing_keywords else "None")

    # -------- CHAT SECTION --------
    question = st.text_input("Ask anything about your document")

    if question:

        docs = vectorstore.similarity_search(question, k=6)
        context = "\n\n".join([doc.page_content for doc in docs])

        API_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-base"

        headers = {
            "Authorization": f"Bearer {os.environ['HUGGINGFACEHUB_API_TOKEN']}"
        }

        history_text = "\n".join(st.session_state.chat_history)

        prompt = f"""
You are an expert resume evaluator and document assistant.

Chat History:
{history_text}

Context:
{context}

Question:
{question}
"""

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 400,
                "temperature": 0.1
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            answer = result[0]["generated_text"]
        else:
            answer = f"API Error: {response.status_code}"

        st.session_state.chat_history.append(f"User: {question}")
        st.session_state.chat_history.append(f"Assistant: {answer}")

        st.subheader("💬 Answer")
        st.write(answer)
