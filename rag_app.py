import streamlit as st
import os
import requests
import tempfile
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Smart Document AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Smart Document AI")
st.caption("Production-Grade Conversational RAG System")

# ==========================================
# SIDEBAR CONTROLS (SaaS style)
# ==========================================

with st.sidebar:
    st.header("⚙️ AI Settings")

    temperature = st.slider(
        "Creativity (Temperature)",
        0.0, 1.0, 0.2, 0.05
    )

    max_tokens = st.slider(
        "Max Response Length",
        100, 800, 300, 50
    )

    st.markdown("---")
    st.markdown("### 📂 Uploaded Documents")

    if "doc_names" in st.session_state:
        for name in st.session_state.doc_names:
            st.write("•", name)

# ==========================================
# CHECK TOKEN
# ==========================================

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("HuggingFace token not found.")
    st.stop()

# ==========================================
# SESSION STATE INIT
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

# ==========================================
# FILE UPLOAD (MULTI DOC)
# ==========================================

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.vector_ready:

    all_text = ""
    doc_names = []

    for uploaded_file in uploaded_files:
        doc_names.append(uploaded_file.name)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            reader = PdfReader(tmp.name)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

    if not all_text.strip():
        st.error("PDF contains no readable text.")
        st.stop()

    @st.cache_resource
    def load_model():
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    model = load_model()

    chunk_size = 800
    overlap = 150
    chunks = []

    for i in range(0, len(all_text), chunk_size - overlap):
        chunk = all_text[i:i+chunk_size]
        if len(chunk.strip()) > 50:
            chunks.append(chunk)

    embeddings = model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.session_state.model = model
    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.vector_ready = True
    st.session_state.doc_names = doc_names

    st.success("Documents processed successfully.")

if not st.session_state.vector_ready:
    st.info("Upload documents to begin.")
    st.stop()

# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# CHAT INPUT
# ==========================================

user_input = st.chat_input("Ask anything about your uploaded documents...")

if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    model = st.session_state.model
    index = st.session_state.index
    chunks = st.session_state.chunks

    # ======================================
    # FAST KEYWORD SEARCH
    # ======================================

    keyword = user_input.strip().lower().replace("?", "")

    for chunk in chunks:
        if keyword in chunk.lower() and len(keyword.split()) <= 2:
            answer = next(
                (line.strip() for line in chunk.split("\n")
                 if keyword in line.lower()),
                None
            )

            if answer:
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.stop()

    # ======================================
    # SEMANTIC SEARCH
    # ======================================

    query_vector = model.encode([user_input])
    D, I = index.search(np.array(query_vector), k=5)

    relevant_chunks = []
    for distance, idx in zip(D[0], I[0]):
        if distance < 1.7:
            relevant_chunks.append(chunks[idx])

    if not relevant_chunks:
        answer = "Not found in document."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.stop()

    context = "\n\n".join(relevant_chunks)

    # ======================================
    # BUILD PROMPT
    # ======================================

    conversation_history = ""
    for msg in st.session_state.messages[-6:]:
        conversation_history += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
You are a professional AI document assistant.

Rules:
- Use only the given context.
- If factual question, answer short and precise.
- If analytical, format clearly.
- If not found, reply exactly:
  Not found in document.

Conversation:
{conversation_history}

Context:
{context}

User:
{user_input}
"""

    # ======================================
    # STREAMING OUTPUT
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
        "max_tokens": max_tokens,
        "stream": True
    }

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with requests.post(API_URL, headers=headers,
                           json=payload, stream=True) as response:

            for line in response.iter_lines():
                if line:
                    decoded = line.decode("utf-8")

                    if decoded.startswith("data: "):
                        data = decoded[6:]

                        if data.strip() == "[DONE]":
                            break

                        try:
                            chunk_json = eval(data)
                            delta = chunk_json["choices"][0]["delta"]
                            content = delta.get("content", "")
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")
                        except:
                            pass

        message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )