import streamlit as st
import tempfile

from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Document Q&A Pro", layout="wide")
st.title("📄 Smart Document Q&A (Local Model Version)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# ---------------- LOAD LOCAL MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

generator = load_model()

# ---------------- DOCUMENT PROCESSING ----------------
@st.cache_resource
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore


# ---------------- MAIN LOGIC ----------------
if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    vectorstore = process_document(tmp_path)
    st.success("Document processed successfully!")

    question = st.text_input("Ask a question about your document")

    if question:

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6}
        )

        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer using ONLY the context below.
If not found, say 'Not found in document'.

Context:
{context}

Question:
{question}
"""

        result = generator(prompt, max_length=512)
        answer = result[0]["generated_text"]

        st.subheader("Answer")
        st.write(answer)
