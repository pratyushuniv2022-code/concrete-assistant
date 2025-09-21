# step2_query_core_qdrant.py
import sys
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ---------------- CONFIG ----------------
QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION = "core_local_384"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 30
# ----------------------------------------

def make_qdrant_client(host, port, retries=4, delay=1.0):
    last_exc = None
    for i in range(retries):
        try:
            client = QdrantClient(host=host, port=port, check_compatibility=False)
            # quick ping
            _ = client.get_collections()
            return client
        except Exception as e:
            last_exc = e
            print(f"[qdrant] connect attempt {i+1}/{retries} failed: {e}")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Qdrant at {host}:{port}. Last error: {last_exc}")

def main():
    # Get query: CLI arg or interactive input
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            return

    print("Loading local embedder:", MODEL_NAME)
    embedder = SentenceTransformer(MODEL_NAME)
    print("Embedding query...")
    qvec = embedder.encode([query])[0].tolist()

    print("Connecting to Qdrant...")
    qc = make_qdrant_client(QDRANT_HOST, QDRANT_PORT)

    # verify collection exists
    cols = [c.name for c in qc.get_collections().collections]
    if COLLECTION not in cols:
        print(f"Collection '{COLLECTION}' not found. Available collections:", cols)
        return

    print(f"Searching collection '{COLLECTION}' (top_k={TOP_K})...")
    results = qc.search(collection_name=COLLECTION, query_vector=qvec, limit=TOP_K, with_payload=True)

    if not results:
        print("No results returned.")
        return

    print("\nTop results:")
    for idx, r in enumerate(results, start=1):
        payload = r.payload or {}
        score = getattr(r, "score", None)
        source = payload.get("source", "unknown")
        page = payload.get("page", "n/a")
        chunk_index = payload.get("chunk_index", "n/a")
        snippet = payload.get("snippet") or payload.get("text") or ""
        print(f"---\n{idx}. score={score:.4f} | source={source} | page={page} | chunk={chunk_index}")
        print(snippet[:800].replace("\n", " ").strip() + ("..." if len(snippet) > 800 else ""))
    print("---\nSearch complete.")

if __name__ == "__main__":
    main()
