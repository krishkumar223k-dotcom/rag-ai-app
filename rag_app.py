import streamlit as st
import tempfile
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart RAG Pro", layout="wide")
st.title("🚀 Smart Document Q&A Pro (Multi-Doc + Resume Analyzer)")

# ---------------- LOAD LOCAL MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large"
    )

llm = load_model()

# ---------------- PROCESS DOCUMENTS ----------------
@st.cache_resource
def process_documents(files):
    all_docs = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250
    )

    split_docs = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

# ---------------- FILE UPLOAD ----------------
uploaded_files = st.file_uploader(
    "Upload one or multiple PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    vectorstore = process_documents(uploaded_files)
    st.success("Documents processed successfully!")

    # Chat memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about your documents")

    if user_question:

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6}
        )

        docs = retriever.get_relevant_documents(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a professional document analyst.

Rules:
- Answer ONLY from context.
- If information exists, extract clearly.
- If not found, say: Not found in document.

Context:
{context}

Question:
{user_question}
"""

        result = llm(prompt, max_length=512)
        answer = result[0]["generated_text"]

        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("AI", answer))

    # Display chat
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 {message}**")
        else:
            st.markdown(f"**🤖 {message}**")

    # ---------------- Resume Analyzer Mode ----------------
    st.markdown("---")
    st.subheader("📄 Resume Analyzer Mode")

    if st.button("Analyze Resume"):

        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10})
        docs = retriever.get_relevant_documents("Extract skills, education, CGPA, and projects")
        context = "\n\n".join([doc.page_content for doc in docs])

        analysis_prompt = f"""
You are a resume evaluator.

Extract:
1. Skills
2. Education
3. CGPA
4. Projects
5. Overall Resume Score out of 10

Context:
{context}
"""

        result = llm(analysis_prompt, max_length=600)
        analysis = result[0]["generated_text"]

        st.markdown("### 📊 Resume Analysis")
        st.write(analysis)
