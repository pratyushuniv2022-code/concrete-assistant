import os
import sys
import time
import uuid
import hashlib
from pathlib import Path
from tqdm import tqdm
from config_env import QDRANT_URL, QDRANT_API_KEY

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ---------------- CONFIG ----------------
CORE_COLL = os.environ.get("CORE_COLL", "core_local_384")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "all-MiniLM-L6-v2")
CORE_PDF_DIR = Path(os.environ.get("CORE_PDF_DIR", "core_pdfs"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", 6))
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", 1.5))

# Qdrant Cloud configuration via environment variables or Streamlit secrets
QDRANT_URL = os.environ.get("QDRANT_URL")      # e.g., "https://<cluster>.qdrant.io"
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # API Key from Qdrant Cloud

if not QDRANT_URL or not QDRANT_API_KEY:
    print("Qdrant Cloud URL or API Key not set. Please set QDRANT_URL and QDRANT_API_KEY.")
    sys.exit(1)

# ---------------- Init ----------------
embedder = SentenceTransformer(LOCAL_MODEL)
VECTOR_DIM = embedder.get_sentence_embedding_dimension()
print("Local embedder dim:", VECTOR_DIM)

# ---------------- Qdrant Cloud helper ----------------
def make_qdrant_client_cloud(url, api_key, retries=RETRY_ATTEMPTS, delay=RETRY_DELAY):
    last_exc = None
    for i in range(retries):
        try:
            client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
            _ = client.get_collections()
            return client
        except Exception as e:
            last_exc = e
            print(f"[qdrant] connect attempt {i+1}/{retries} failed: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Qdrant at {url}. Last error: {last_exc}")

qdrant = make_qdrant_client_cloud(QDRANT_URL, QDRANT_API_KEY)

# Ensure collection exists
existing = [c.name for c in qdrant.get_collections().collections]
if CORE_COLL not in existing:
    print(f"Creating collection '{CORE_COLL}' with dim={VECTOR_DIM}")
    qdrant.create_collection(
        collection_name=CORE_COLL,
        vectors_config=qm.VectorParams(size=VECTOR_DIM, distance=qm.Distance.COSINE)
    )
else:
    print(f"Collection '{CORE_COLL}' already exists.")

# ---------------- PDF helpers ----------------
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

# ---------------- Process core PDFs ----------------
CORE_PDF_DIR.mkdir(exist_ok=True)
pdf_files = list(CORE_PDF_DIR.glob("*.pdf"))
if not pdf_files:
    print("No PDFs found in core_pdfs/. Put your PDF files there and rerun.")
    sys.exit(1)

texts = []
metadatas = []

for pdf_file in pdf_files:
    print("Processing:", pdf_file.name)
    pages = extract_pages(pdf_file)
    for pg in pages:
        if not pg["text"].strip():
            continue
        chunks = chunk_text(pg["text"])
        for idx, c in enumerate(chunks):
            stable = hashlib.sha1(f"{pdf_file.name}-{pg['page']}-{idx}".encode()).hexdigest()
            texts.append(c)
            metadatas.append({
                "source": pdf_file.name,
                "page": pg["page"],
                "chunk_index": idx,
                "type": "core",
                "stable_id": stable,
                "snippet": c[:1000]
            })

total = len(texts)
print(f"Total chunks to embed: {total}")
if total == 0:
    print("No text chunks found - exiting.")
    sys.exit(0)

# ---------------- Batch embedding & upsert ----------------
def batch_iter(items, n):
    for i in range(0, len(items), n):
        yield i, items[i:i+n]

for i, batch_texts in batch_iter(texts, BATCH_SIZE):
    batch_idx = i // BATCH_SIZE + 1
    batch_vectors = embedder.encode(batch_texts, show_progress_bar=False)
    batch_payloads = metadatas[i:i+len(batch_texts)]
    batch_ids = [str(uuid.uuid4()) for _ in batch_texts]
    try:
        qdrant.upsert(
            collection_name=CORE_COLL,
            points=qm.Batch(ids=batch_ids, vectors=batch_vectors.tolist(), payloads=batch_payloads)
        )
        print(f"Upserted batch {batch_idx} (items {i+1}-{i+len(batch_texts)})")
    except Exception as e:
        print("Upsert failed on batch", batch_idx, "error:", e)
        raise

# ---------------- Verification ----------------
count_resp = qdrant.count(CORE_COLL)
count = getattr(count_resp, "count", None)
print("Indexed vector count in collection (reported):", count)

# ---------------- Sample search ----------------
sample_q = "What are typical constituents of concrete mixture?"
qvec = embedder.encode([sample_q])[0].tolist()
hits = qdrant.search(collection_name=CORE_COLL, query_vector=qvec, limit=3, with_payload=True)
print("\nTop hits (sample):")
for h in hits:
    print("score:", h.score)
    p = h.payload
    print(p.get("source"), "pg", p.get("page"), "chunk_index", p.get("chunk_index"))
    print(p.get("snippet")[:400].replace("\n", " "), "...\n")