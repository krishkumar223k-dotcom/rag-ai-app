import streamlit as st
import os
import requests
import tempfile
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(page_title="Smart Document AI", layout="wide")
st.title("📄 Smart Document AI (Multi-PDF + Accurate Version)")

# ==============================
# CHECK TOKEN
# ==============================

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if HF_TOKEN:
    st.success("Token Loaded: True")
else:
    st.error("Token not found! Check Streamlit Secrets.")
    st.stop()

# ==============================
# UPLOAD MULTIPLE PDFs
# ==============================

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

all_text = ""

for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        reader = PdfReader(tmp.name)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

st.success("Documents processed successfully!")

# ==============================
# EMBEDDINGS + FAISS
# ==============================

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks = [all_text[i:i+800] for i in range(0, len(all_text), 800)]

embeddings = model.encode(chunks)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ==============================
# MODE SELECTION
# ==============================

mode = st.selectbox(
    "Choose Mode",
    ["Ask Question", "Resume Scoring (1–10)", "ATS Match"]
)

user_input = st.text_area("Enter your question or job description")

if not user_input:
    st.stop()

# ==============================
# RETRIEVAL
# ==============================

query_vector = model.encode([user_input])
D, I = index.search(np.array(query_vector), k=5)

retrieved_context = "\n\n".join([chunks[i] for i in I[0]])

# ==============================
# PROMPT LOGIC
# ==============================

if mode == "Ask Question":
    prompt = f"""
Answer strictly using the context below.
If answer not present, say: Not found in document.

Context:
{retrieved_context}

Question:
{user_input}
"""

elif mode == "Resume Scoring (1–10)":
    prompt = f"""
You are a professional resume evaluator.

Based on the resume below, give:
1. Resume score from 1 to 10
2. Strengths
3. Weaknesses
4. Suggestions to improve

Resume:
{retrieved_context}
"""

elif mode == "ATS Match":
    prompt = f"""
Compare this resume with the job description.

Return:
1. ATS match percentage
2. Missing keywords
3. Improvement suggestions

Resume:
{retrieved_context}

Job Description:
{user_input}
"""

# ==============================
# HUGGING FACE ROUTER CALL
# ==============================

API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "model": "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.3,
    "max_tokens": 700
}

response = requests.post(API_URL, headers=headers, json=payload)

# ==============================
# DEBUG OUTPUT
# ==============================

if response.status_code == 200:
    result = response.json()
    answer = result["choices"][0]["message"]["content"]
else:
    answer = f"API Error: {response.status_code}\n{response.text}"

# ==============================
# DISPLAY RESULT
# ==============================

st.subheader("Result")
st.write(answer)
