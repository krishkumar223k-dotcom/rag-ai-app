import streamlit as st
import os
import requests
import tempfile
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(page_title="Smart Document AI", layout="wide")

st.title("🧠 Smart Document AI")
st.caption("ChatGPT-style Conversational RAG System")

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "Chat"

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("## 🧠 Smart AI")

    if st.button("➕ New Chat"):
        st.session_state.messages = []

    st.markdown("---")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")
    st.markdown("### 🛠 Tools")

    if st.button("💬 Chat Mode"):
        st.session_state.mode = "Chat"

    if st.button("📄 Resume Scoring (1–10)"):
        st.session_state.mode = "Resume"

    if st.button("🎯 ATS Match Analysis"):
        st.session_state.mode = "ATS"

    st.markdown("---")
    st.markdown("### ⚙️ AI Settings")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max Tokens", 100, 800, 300, 50)

# ==========================================
# CHECK TOKEN
# ==========================================

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("HuggingFace token missing in Secrets.")
    st.stop()

# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_files:

    full_text = ""

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            reader = PdfReader(tmp.name)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

    # Chunking
    chunks = [full_text[i:i+800] for i in range(0, len(full_text), 800)]

    # Embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = model.encode(chunks)

    if len(embeddings) == 0:
        st.warning("No readable text found in PDF.")
        st.stop()

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

else:
    st.info("Upload PDF to begin.")
    st.stop()

# ==========================================
# CHAT INTERFACE
# ==========================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask anything about your document...")

if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # ======================================
    # HYBRID RETRIEVAL
    # ======================================

    query_vector = model.encode([user_input])
    D, I = index.search(np.array(query_vector), k=5)

    semantic_context = "\n\n".join([chunks[i] for i in I[0]])

    # Simple keyword retrieval
    keyword_hits = []
    for chunk in chunks:
        if user_input.lower() in chunk.lower():
            keyword_hits.append(chunk)

    keyword_context = "\n\n".join(keyword_hits[:3])

    context = semantic_context + "\n\n" + keyword_context

    # ======================================
    # PROMPT BUILDING
    # ======================================

    mode = st.session_state.mode

    if mode == "Chat":

        prompt = f"""
You are a professional AI assistant.

Use only the context below.
If short factual question (like CGPA), answer directly.
If not found, reply exactly:
Not found in document.

Context:
{context}

Question:
{user_input}
"""

    elif mode == "Resume":

        prompt = f"""
You are a professional resume evaluator.

Give:
1. Resume Score (1–10)
2. Strengths
3. Weaknesses
4. Improvement Suggestions

Resume:
{context}
"""

    elif mode == "ATS":

        prompt = f"""
Compare resume with job description.

Return:
1. ATS Match Percentage
2. Missing Keywords
3. Suggestions

Resume:
{context}

Job Description:
{user_input}
"""

    # ======================================
    # API CALL
    # ======================================

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

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        answer = result["choices"][0]["message"]["content"]
    else:
        answer = f"API Error: {response.status_code}"

    # ======================================
    # DISPLAY ANSWER
    # ======================================

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})