# streamlit_concrete_assistant_fast.py
# Importing necessary library
import streamlit as st
from pathlib import Path
import uuid
import hashlib
import time
from config_env import QDRANT_URL, QDRANT_API_KEY, CORE_COLL, LOCAL_MODEL, TOP_K
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from PyPDF2 import PdfReader
from step3_query_generate_gemini import generate_answer_conversational

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Concrete Assistant 💬", layout="wide")
st.title("Concrete Assistant 💬")
st.write("Ask questions about concrete or upload your own PDFs for context.")

# ---------------- Embedder ----------------
embedder = SentenceTransformer(
    LOCAL_MODEL, 
    use_auth_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
)

# ---------------- Lazy Qdrant client ----------------
_qdrant = None
def get_qdrant_client():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)
        VECTOR_DIM = embedder.get_sentence_embedding_dimension()
        existing = [c.name for c in _qdrant.get_collections().collections]
        if CORE_COLL not in existing:
            _qdrant.create_collection(
                collection_name=CORE_COLL,
                vectors_config=qm.VectorParams(size=VECTOR_DIM, distance=qm.Distance.COSINE)
            )
    return _qdrant

# ---------------- PDF Helpers ----------------
def extract_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, p in enumerate(reader.pages):
        txt = p.extract_text() or ""
        pages.append({"page": i+1, "text": txt})
    return pages

def chunk_text(text, chunk_size=600, overlap=200):
    words = text.split()
    chunks = []
    i = 0
    if not words:
        return []
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def index_pdf_to_qdrant(pdf_path, collection_name, doc_type="user_pdf"):
    qdrant = get_qdrant_client()
    pages = extract_pages(pdf_path)
    texts, metadatas = [], []
    for pg in pages:
        chunks = chunk_text(pg["text"])
        for idx, c in enumerate(chunks):
            stable = hashlib.sha1(f"{pdf_path.name}-{pg['page']}-{idx}".encode()).hexdigest()
            texts.append(c)
            metadatas.append({
                "source": pdf_path.name,
                "page": pg["page"],
                "chunk_index": idx,
                "type": doc_type,
                "stable_id": stable,
                "snippet": c[:1000]
            })
    if texts:
        vectors = embedder.encode(texts, show_progress_bar=True)
        point_ids = [str(uuid.uuid4()) for _ in texts]
        qdrant.upsert(
            collection_name=collection_name,
            points=qm.Batch(ids=point_ids, vectors=vectors.tolist(), payloads=metadatas)
        )
        st.success(f"Indexed {pdf_path.name} into Qdrant collection '{collection_name}'.")

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []

UPLOAD_DIR = Path("user_uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------- Core PDFs ----------------
CORE_PDF_DIR = Path("core_pdfs")
CORE_PDF_DIR.mkdir(exist_ok=True)
pdf_files = list(CORE_PDF_DIR.glob("*.pdf"))
with st.expander("📄 View/Download Core PDFs"):
    if pdf_files:
        for pdf_file in pdf_files:
            with open(pdf_file, "rb") as f:
                st.download_button(f"Download {pdf_file.name}", data=f, file_name=pdf_file.name)
    else:
        st.info("No core PDFs found. Place them in the 'core_pdfs' folder.")

# Pre-index core PDFs (once)
if st.button("🔹 Pre-embed Core PDFs (fast startup)"):
    for pdf_file in pdf_files:
        index_pdf_to_qdrant(pdf_file, CORE_COLL, doc_type="core")
    st.success("Core PDFs pre-embedded!")

# ---------------- Upload PDFs ----------------
uploaded_pdf = st.file_uploader("Upload your PDF for Q&A", type="pdf")
if uploaded_pdf:
    pdf_path = UPLOAD_DIR / uploaded_pdf.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
    st.success(f"Uploaded {uploaded_pdf.name} successfully!")
    index_pdf_to_qdrant(pdf_path, CORE_COLL, doc_type="user_pdf")

# ---------------- Ask Question ----------------
user_question = st.text_input("Ask me about concrete:")
if user_question:
    qdrant = get_qdrant_client()
    
    # Embed only the user question
    query_vector = embedder.encode([user_question])[0].tolist()
    
    # Search top-k relevant chunks
    results = qdrant.search(
        collection_name=CORE_COLL,
        query_vector=query_vector,
        limit=TOP_K,
        with_payload=True
    )
    
    # Prepare minimal context for LLM
    top_chunks = results[:3]  # take only top 3
    context = "\n".join([h.payload["snippet"] for h in top_chunks])
    
    # Generate answer
    start_time = time.time()
    answer = generate_answer_conversational(user_question, st.session_state.history, context=context)
    elapsed = time.time() - start_time
    st.markdown(f"**Assistant:** {answer}")
    st.write(f"*Response time: {elapsed:.2f}s*")
    st.session_state.history.append({"user": user_question, "assistant": answer})

# ---------------- Conversation History ----------------
if st.checkbox("Show conversation history"):
    for turn in st.session_state.history:
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Assistant:** {turn['assistant']}")
        st.write("---")
