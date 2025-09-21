# streamlit_concrete_assistant_cloud.py
#importing necessary packages
import streamlit as st
from pathlib import Path
import uuid
import hashlib
import time

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# Our existing modules
from step1_index_core_qdrant import embedder, chunk_text, extract_pages
from step3_query_generate_gemini import generate_answer_conversational  # using LLM backend

# ---------------- CONFIG ----------------
CORE_COLLECTION = "core_local_384"
USER_COLLECTION = "user_pdfs"
VECTOR_DIM = embedder.get_sentence_embedding_dimension()
UPLOAD_DIR = Path("user_uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

TOP_K = 15
TRUNCATE_SNIPPET = 1200  # max chars per snippet

# Core PDFs (optional)
CORE_PDF_DIR = Path("core_pdfs")
CORE_PDF_DIR.mkdir(exist_ok=True)
PDF_URLS = {pdf.name: pdf for pdf in CORE_PDF_DIR.glob("*.pdf")}

# ---------------- Initialize Qdrant Cloud ----------------
qdrant = QdrantClient(
    url=st.secrets["QDRANT"]["url"],       # from Streamlit secrets
    api_key=st.secrets["QDRANT"]["api_key"]
)

# Ensure USER_COLLECTION exists
existing_collections = [c.name for c in qdrant.get_collections().collections]
if USER_COLLECTION not in existing_collections:
    qdrant.create_collection(
        collection_name=USER_COLLECTION,
        vectors_config=qm.VectorParams(size=VECTOR_DIM, distance=qm.Distance.COSINE)
    )

# ---------------- Streamlit UI ----------------
st.title("Concrete Assistant 💬")
st.write("Ask questions about concrete or upload your own PDFs for context.")

# Section: Core PDFs download
with st.expander("📄 View/Download Original PDFs"):
    if PDF_URLS:
        for name, path in PDF_URLS.items():
            with open(path, "rb") as f:
                st.download_button(
                    label=f"Download {name}",
                    data=f,
                    file_name=name,
                    mime="application/pdf"
                )
    else:
        st.info("No core PDFs found. Place them in the 'core_pdfs' folder.")

# Session state for conversation
if "history" not in st.session_state:
    st.session_state.history = []

# PDF Upload
uploaded_pdf = st.file_uploader("Upload your PDF to include in Q&A", type="pdf")
if uploaded_pdf:
    pdf_path = UPLOAD_DIR / uploaded_pdf.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success(f"Uploaded {uploaded_pdf.name} successfully!")

    # Index PDF to Qdrant
    pages = extract_pages(pdf_path)
    texts, metadatas = [], []
    for pg in pages:
        chunks = chunk_text(pg["text"], chunk_size=600, overlap=200)
        for idx, c in enumerate(chunks):
            stable = hashlib.sha1(f"{pdf_path.name}-{pg['page']}-{idx}".encode()).hexdigest()
            texts.append(c)
            metadatas.append({
                "source": pdf_path.name,
                "page": pg["page"],
                "chunk_index": idx,
                "type": "user_pdf",
                "stable_id": stable,
                "snippet": c[:1000]
            })

    vectors = embedder.encode(texts, show_progress_bar=True)
    point_ids = [str(uuid.uuid4()) for _ in texts]
    qdrant.upsert(
        collection_name=USER_COLLECTION,
        points=qm.Batch(ids=point_ids, vectors=vectors.tolist(), payloads=metadatas)
    )
    st.info(f"Indexed {uploaded_pdf.name} into Qdrant collection '{USER_COLLECTION}'.")

# User question input
user_question = st.text_input("Ask me about concrete:")
if user_question:
    start_time = time.time()
    answer = generate_answer_conversational(user_question, st.session_state.history)
    elapsed = time.time() - start_time

    st.markdown(f"**Assistant:** {answer}")
    st.write(f"*Response time: {elapsed:.2f}s*")

    st.session_state.history.append({"user": user_question, "assistant": answer})

# Optionally show conversation history
if st.checkbox("Show conversation history"):
    for turn in st.session_state.history:
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Assistant:** {turn['assistant']}")
        st.write("---")
