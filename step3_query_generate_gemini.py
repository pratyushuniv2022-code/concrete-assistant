# step3_query_generate_gemini_vertexai_conversational.py
#importing necessary packages
import os
from pathlib import Path
import sys
import time
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from qdrant_client import QdrantClient

# ---------------- CONFIG ----------------
PROJECT_ID = "sigma-icon-472417-c4"
LOCATION = "us-central1"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "127.0.0.1")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
# QDRANT_HOST = "127.0.0.1"
# QDRANT_PORT = 6333
COLLECTION_NAME = "core_local_384"  # 384-d vector collection
TOP_K = 20
TRUNCATE_SNIPPET = 1200  # max chars per snippet
KEYWORD_BOOST = [
    # Cement types and admixtures
    "blended cement", "OPC", "ordinary portland cement", "fly-ash", "slag", "silica fume", "mineral admixtures",
    
    # Concrete properties
    "compressive strength", "tensile strength", "durability", "workability", "permeability", "consistency", "setting time", "hardening", "curing",
    
    # Concrete mix and design
    "water-cement ratio", "w/c ratio", "mix design", "proportioning", "grading", "aggregate", "fine aggregate", "coarse aggregate",
    
    # Concrete compaction
    "vibrator", "internal vibrator", "external vibrator", "needle vibrator", "surface vibrator", "vibrating table", "vibratory screed", "compaction",
    
    # Thermal and environmental aspects
    "temperature", "heat of hydration", "energy saving", "pollution control", "environmental advantages", "sustainability",
    
    # Strength & durability
    "cracking", "shrinkage", "creep", "load-bearing", "toughness", "modulus of elasticity"
]


# ---------------- Initialize Clients ----------------
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1")
)

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

try:
    _ = qdrant_client.get_collections()
except Exception as e:
    print("ERROR: Cannot connect to Qdrant:", e)
    sys.exit(1)

# ---------------- Helpers ----------------
def get_embedding_local(text, model_dim=384):
    """Generate embeddings from Qdrant local collection (already indexed)."""
    # In this setup, you already have 384-d vectors, so we just return a dummy vector for search
    # Actual query will use Qdrant collection vectors
    return [0.0] * model_dim

def retrieve_passages(question, top_k=TOP_K):
    """Retrieve top-k passages from Qdrant collection."""
    # Here we use your local embeddings stored in Qdrant
    query_vector = get_embedding_local(question)
    try:
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
    except Exception:
        # fallback to deprecated search if query_points not available
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
    passages = []
    for hit in results:
        payload = hit.payload or {}
        snippet = payload.get("snippet") or payload.get("text") or ""
        source = payload.get("source", "unknown")
        page = payload.get("page", -1)
        score = getattr(hit, "score", 0.0)
        passages.append({
            "text": snippet,
            "source": source,
            "page": page,
            "score": float(score)
        })
    return passages

def hybrid_rank(passages, question):
    """Rank passages by original score + keyword boosting."""
    q_lower = question.lower()
    def boosted(p):
        score = p.get("score", 0.0)
        txt = (p.get("text") or "").lower()
        boost = sum(0.15 for kw in KEYWORD_BOOST if kw in txt or kw in q_lower)
        return score + boost
    return sorted(passages, key=boosted, reverse=True)

def build_prompt(current_question, conversation_history, passages):
    """Build prompt for Gemini LLM with conversation history and numbered passages."""
    history_text = ""
    for turn in conversation_history[-10:]:
        if isinstance(turn, dict):
            history_text += f"You: {turn.get('user','')}\nAssistant: {turn.get('assistant','')}\n"
        else:
            history_text += f"{turn}\n"

    numbered = []
    for i, p in enumerate(passages, start=1):
        snippet = (p["text"] or "").strip()
        if len(snippet) > TRUNCATE_SNIPPET:
            snippet = snippet[:TRUNCATE_SNIPPET] + "..."
        numbered.append(f"[{i}] Source: {p['source']}, Page: {p['page']}\n{snippet}")
    context_block = "\n\n".join(numbered)

    prompt = f"""{history_text}
You are given numbered passages extracted from technical documents (provenance included).
Answer the user's question using ONLY the passages below. If exact answer is not present, say "I don't know" and list relevant passage numbers.

Passages:
{context_block}

Question:
{current_question}

Instructions:
- Use ONLY the passages above; do not invent facts.
- Provide a structured, complete answer in 3-6 sentences per section.
- Cite passage numbers in square brackets, e.g., (see [1], [3]).
- If more details are available, summarize all points concisely.
"""
    return prompt

def generate_answer_conversational(current_question, conversation_history, top_chunks=None):
    """
    Generate answer using Gemini LLM with conversation context.
    Optionally pass top_chunks retrieved from Qdrant to reduce latency.
    """
    # Build minimal context from retrieved Qdrant passages
    passages = []
    if top_chunks:
        for hit in top_chunks:
            payload = hit.payload or {}
            passages.append({
                "text": payload.get("snippet") or payload.get("text") or "",
                "source": payload.get("source", "unknown"),
                "page": payload.get("page", -1),
                "score": getattr(hit, "score", 0.0)
            })
    ranked_passages = hybrid_rank(passages, current_question)
    prompt = build_prompt(current_question, conversation_history, ranked_passages)

    cfg = GenerateContentConfig(temperature=0.2, max_output_tokens=2000)
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=cfg
    )
    return getattr(resp, "text", str(resp))

# ---------------- Demo ----------------
if __name__ == "__main__":
    conversation_history = []
    print("Enter 'exit' to quit.\n")
    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in ["exit", "quit"]:
            break
        # Fetch top-k from Qdrant
        query_vector = [0.0]*384  # dummy placeholder; replace with embeddings if available
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=TOP_K,
            with_payload=True
        )
        answer = generate_answer_conversational(user_q, conversation_history, top_chunks=results[:3])
        print("\nAssistant:", answer, "\n")
        conversation_history.append({"user": user_q, "assistant": answer})
