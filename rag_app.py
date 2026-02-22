import streamlit as st
import os
import requests
import tempfile
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(page_title="Smart Document AI", layout="wide")
st.title("📄 Smart Document AI (Production Version)")
st.caption("RAG + Resume Scoring + ATS Matching")

# =====================================
# CHECK TOKEN
# =====================================

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("HuggingFace token not found in Secrets.")
    st.stop()

# =====================================
# FILE UPLOAD
# =====================================

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload at least one PDF to begin.")
    st.stop()

# =====================================
# EXTRACT TEXT
# =====================================

all_text = ""

for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        reader = PdfReader(tmp.name)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

if len(all_text.strip()) == 0:
    st.error("❌ This PDF contains no extractable text (possibly scanned).")
    st.stop()

st.success("Documents processed successfully.")

# =====================================
# LOAD EMBEDDING MODEL (Cached)
# =====================================

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# =====================================
# SMART CHUNKING (OVERLAP)
# =====================================

chunk_size = 800
overlap = 150

chunks = []

for i in range(0, len(all_text), chunk_size - overlap):
    chunk = all_text[i:i+chunk_size]
    if len(chunk.strip()) > 50:
        chunks.append(chunk)

if len(chunks) == 0:
    st.error("Could not create meaningful text chunks.")
    st.stop()

# =====================================
# VECTOR EMBEDDING + FAISS
# =====================================

embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# =====================================
# MODE SELECTION
# =====================================

mode = st.selectbox(
    "Choose Mode",
    ["Ask Question", "Resume Scoring (1–10)", "ATS Match"]
)

# =====================================
# MOBILE FRIENDLY FORM
# =====================================

with st.form("query_form"):
    user_input = st.text_area("Enter your question or job description", height=150)
    submitted = st.form_submit_button("🚀 Generate")

if not submitted:
    st.stop()

if not user_input.strip():
    st.warning("Please enter something.")
    st.stop()

# =====================================
# SEMANTIC RETRIEVAL (FILTERED)
# =====================================

query_vector = model.encode([user_input])
D, I = index.search(np.array(query_vector), k=3)

threshold = 1.2
relevant_chunks = []

for distance, idx in zip(D[0], I[0]):
    if distance < threshold:
        relevant_chunks.append(chunks[idx])

if len(relevant_chunks) == 0:
    st.warning("Not found in document.")
    st.stop()

retrieved_context = "\n\n".join(relevant_chunks)

# =====================================
# STRICT PROMPT
# =====================================

if mode == "Ask Question":

    prompt = f"""
You are a strict document assistant.

Rules:
- Answer ONLY using the context below.
- Do NOT use outside knowledge.
- If answer is not clearly present, reply exactly:
  Not found in document.
- Do not guess.

Context:
{retrieved_context}

Question:
{user_input}
"""

elif mode == "Resume Scoring (1–10)":

    prompt = f"""
You are a professional resume evaluator.

Provide:
1. Resume score (1–10)
2. Strengths
3. Weaknesses
4. Suggestions

Resume:
{retrieved_context}
"""

elif mode == "ATS Match":

    prompt = f"""
Compare this resume with the job description.

Return:
1. ATS match percentage
2. Missing keywords
3. Suggestions

Resume:
{retrieved_context}

Job Description:
{user_input}
"""

# =====================================
# HUGGING FACE ROUTER CALL
# =====================================

API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "model": "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.1,  # low temperature = less hallucination
    "max_tokens": 700
}

with st.spinner("Generating response..."):
    response = requests.post(API_URL, headers=headers, json=payload)

st.subheader("📌 Result")

if response.status_code == 200:
    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    st.success("Response Generated Successfully")
    st.write(answer)

    # =====================================
    # PDF DOWNLOAD
    # =====================================

    def generate_pdf(text):
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Smart Document AI Result", styles["Heading1"]))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Preformatted(text, styles["Code"]))

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_file = generate_pdf(answer)

    st.download_button(
        label="📥 Download as PDF",
        data=pdf_file,
        file_name="Smart_Document_AI_Result.pdf",
        mime="application/pdf"
    )

else:
    st.error(f"API Error: {response.status_code}")
    st.code(response.text)