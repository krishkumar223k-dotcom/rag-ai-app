import streamlit as st
import os
import requests
import tempfile
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Preformatted
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import io

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Smart Document AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart Document AI")
st.caption("Multi-PDF • Resume Analyzer • ATS Matching • Production Version")

# ==============================
# LOAD TOKEN
# ==============================

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("❌ HuggingFace token not found in Secrets.")
    st.stop()

# ==============================
# SIDEBAR SETTINGS
# ==============================

st.sidebar.header("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Ask Question", "Resume Scoring (1–10)", "ATS Match"]
)

temperature = st.sidebar.slider("Creativity", 0.0, 1.0, 0.3)
max_tokens = st.sidebar.slider("Max Response Length", 200, 1500, 700)

# ==============================
# FILE UPLOAD
# ==============================

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload PDFs to begin.")
    st.stop()

# ==============================
# EXTRACT TEXT
# ==============================

all_text = ""

for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        reader = PdfReader(tmp.name)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

st.success("Documents processed successfully.")

# ==============================
# EMBEDDINGS + VECTOR SEARCH
# ==============================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_embedding_model()

chunks = [all_text[i:i+800] for i in range(0, len(all_text), 800)]
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ==============================
# INPUT FORM
# ==============================

with st.form("query_form"):
    user_input = st.text_area(
        "Enter your question or job description",
        height=150
    )
    submitted = st.form_submit_button("🚀 Generate")

if not submitted:
    st.stop()

if not user_input.strip():
    st.warning("Please enter something.")
    st.stop()

# ==============================
# RETRIEVE CONTEXT
# ==============================

query_vector = model.encode([user_input])
D, I = index.search(np.array(query_vector), k=5)
retrieved_context = "\n\n".join([chunks[i] for i in I[0]])

# ==============================
# PROMPT BUILDING
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

# ==============================
# HUGGINGFACE ROUTER CALL
# ==============================

API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "model": "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": temperature,
    "max_tokens": max_tokens
}

with st.spinner("Generating response..."):
    response = requests.post(API_URL, headers=headers, json=payload)

# ==============================
# DISPLAY RESULT
# ==============================

st.subheader("📌 Result")

if response.status_code == 200:
    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    st.success("Response Generated Successfully")
    st.markdown(answer)

    # ==============================
    # PDF DOWNLOAD FEATURE
    # ==============================

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
