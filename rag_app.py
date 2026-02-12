import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import tempfile

st.set_page_config(page_title="Smart Document Q&A", layout="wide")
st.title("📄 Smart Document Q&A System (FAST RAG + SOURCES)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


# ---------- CACHE DOCUMENT PROCESSING ----------
@st.cache_resource
def process_document(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


# ---------- MAIN ----------
if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Processing document..."):
        vectorstore = process_document(tmp_path)

    st.success("Document processed successfully!")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = Ollama(model="tinyllama", keep_alive=True)

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY using the provided context.
If answer not found, say "Not in document".

Context:
{context}

Question:
{input}
""")

    query = st.text_input("Ask a question about your document")

    if query:
        with st.spinner("Generating answer..."):

            docs = retriever.invoke(query)

            context = "\n\n".join([d.page_content for d in docs])

            response = llm.invoke(prompt.format(
                context=context,
                input=query
            ))

        st.subheader("Answer:")
        st.write(response)

        st.subheader("Sources:")

        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "Unknown")
            snippet = doc.page_content[:200]

            st.markdown(f"**Source {i} — Page {page}**")
            st.write(snippet + "...")

