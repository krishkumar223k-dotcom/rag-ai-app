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
st.title("📄 Smart Document AI")
st.caption("Hybrid RAG • Resume Scoring • ATS Matching • Clean GPT Style")

# =====================================
# CHECK TOKEN
# =====================================

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("HuggingFace token not found.")
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
    st.error("This PDF contains no extractable text.")
    st.stop()

st.success("Documents processed successfully.")

# =====================================
# LOAD EMBEDDING MODEL
# =====================================

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# =====================================
# SMART CHUNKING WITH OVERLAP
# =====================================

chunk_size = 800
overlap = 150

chunks = []
for i in range(0, len(all_text), chunk_size - overlap):
    chunk = all_text[i:i+chunk_size]
    if len(chunk.strip()) > 50:
        chunks.append(chunk)

if len(chunks) == 0:
    st.error("Could not create meaningful chunks.")
    st.stop()

# =====================================
# VECTOR STORE
# =====================================

embeddings = model.encode(chunks)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# =====================================
# MODE
# =====================================

mode = st.selectbox(
    "Choose Mode",
    ["Ask Question", "Resume Scoring (1–10)", "ATS Match"]
)

# =====================================
# INPUT FORM
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
# FAST KEYWORD SEARCH (FIXES CGPA ISSUE)
# =====================================

if mode == "Ask Question":
    keyword = user_input.strip().lower().replace("?", "")

    if len(keyword.split()) <= 2:
        if keyword in all_text.lower():
            lines = all_text.split("\n")
            for line in lines:
                if keyword in line.lower():
                    st.subheader("📌 Result")
                    clean_line = line.strip()
                    st.success("Response Generated Successfully")
                    st.write(clean_line)
                    st.stop()

# =====================================
# ADAPTIVE SEMANTIC RETRIEVAL
# =====================================

query_vector = model.encode([user_input])

if len(user_input.split()) <= 2:
    k_value = 6
    threshold = 1.8
else:
    k_value = 4
    threshold = 1.5

D, I = index.search(np.array(query_vector), k=k_value)

relevant_chunks = []
for distance, idx in zip(D[0], I[0]):
    if distance < threshold:
        relevant_chunks.append(chunks[idx])

if len(relevant_chunks) == 0:
    st.warning("Not found in document.")
    st.stop()

retrieved_context = "\n\n".join(relevant_chunks)

# =====================================
# STRICT GPT STYLE PROMPT
# =====================================

if mode == "Ask Question":

    prompt = f"""
You are a precise document extraction assistant.

Rules:
- Return ONLY the direct answer.
- Do NOT explain.
- Do NOT add extra text.
- If answer not present, reply exactly:
  Not found in document.

Context:
{retrieved_context}

Question:
{user_input}

Return only the final answer.
"""

elif mode == "Resume Scoring (1–10)":

    prompt = f"""
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
# HUGGING FACE API CALL
# =====================================

API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "model": "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.1,
    "max_tokens": 200
}

with st.spinner("Generating response..."):
    response = requests.post(API_URL, headers=headers, json=payload)

st.subheader("📌 Result")

if response.status_code == 200:
    result = response.json()
    answer = result["choices"][0]["message"]["content"].strip()

    # Clean extra lines
    answer = answer.split("\n")[0]

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