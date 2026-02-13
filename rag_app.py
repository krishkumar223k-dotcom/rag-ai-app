import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser


# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Smart Document Q&A", layout="wide")
st.title("📄 Smart Document Q&A System (Cloud Version)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


@st.cache_resource
def process_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    return vectorstore


if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    vectorstore = process_document(tmp_path)
    st.success("Document processed successfully!")

    question = st.text_input("Ask a question about your document")

    if question:

        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-base",
            temperature=0,
            max_new_tokens=512,
            huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )

        final_prompt = f"""
Answer the question using ONLY the context below.
If the answer is not found, say 'Not found in document'.

Context:
{context}

Question:
{question}
"""

        parser = StrOutputParser()
        chain = llm | parser

        response = chain.invoke(final_prompt)

        st.subheader("Answer")
        st.write(response)
