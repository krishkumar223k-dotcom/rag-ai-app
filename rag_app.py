import streamlit as st
import os
import requests
import tempfile
import json
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Smart Document AI",
    page_icon="🧠",
    layout="wide"
)

# =========================================
# CUSTOM CHATGPT-LIKE CSS
# =========================================

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    max-width: 900px;
    margin: auto;
}

section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}

.stChatMessage {
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-size: 15px;
}

[data-testid="stChatMessage-user"] {
    background-color: #1e293b;
}

[data-testid="stChatMessage-assistant"] {
    background-color: #111827;
}

textarea {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# CHECK TOKEN
# =========================================

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    st.error("HuggingFace token missing.")
    st.stop()

# =========================================
# SESSION STATE INIT
# =========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False

if "chunks" not in st.session_state:
    st.session_state.chunks = []

# =========================================
# SIDEBAR
# =========================================

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

    st.markdown("### ⚙️ Settings")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max Tokens", 100, 800, 300, 50)

# =========================================
# DOCUMENT PROCESSING
# =========================================

if uploaded_files and not st.session_state.vector_ready:

    all_text = ""

    for uploaded_file in uploaded_files:

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

    st.success("Documents processed successfully.")

# Stop if no docs
if not st.session_state.vector_ready:
    st.info("Upload PDF(s) to begin.")
    st.stop()

# =========================================
# HEADER
# =========================================

st.markdown("""
<h2 style='text-align:center;'>Smart Document AI</h2>
<p style='text-align:center; color:gray;'>
ChatGPT-style Conversational RAG System
</p>
""", unsafe_allow_html=True)

# =========================================
# DISPLAY CHAT HISTORY
# =========================================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================================
# CHAT INPUT
# =========================================

user_input = st.chat_input("Message Smart AI...")

if user_input:

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    model = st.session_state.model
    index = st.session_state.index
    chunks = st.session_state.chunks

    # =====================================
    # FAST KEYWORD SEARCH
    # =====================================

    keyword = user_input.lower().replace("?", "").strip()

    if len(keyword.split()) <= 2:
        for chunk in chunks:
            if keyword in chunk.lower():
                for line in chunk.split("\n"):
                    if keyword in line.lower():
                        answer = line.strip()
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                        st.stop()

    # =====================================
    # SEMANTIC SEARCH
    # =====================================

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

    # =====================================
    # BUILD PROMPT
    # =====================================

    prompt = f"""
You are a professional AI document assistant.

Rules:
- Use only provided context.
- Answer clearly and precisely.
- If short factual question, give direct answer only.
- If not found, reply exactly:
  Not found in document.

Context:
{context}

User:
{user_input}
"""

    # =====================================
    # STREAMING RESPONSE (FIXED)
    # =====================================

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

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            stream=True
        )

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")

                if decoded_line.startswith("data: "):
                    data = decoded_line.replace("data: ", "")

                    if data.strip() == "[DONE]":
                        break

                    try:
                        json_data = json.loads(data)
                        delta = json_data["choices"][0]["delta"]

                        if "content" in delta:
                            content = delta["content"]
                            full_response += content
                            message_placeholder.markdown(full_response + "▌")

                    except:
                        pass

        message_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )