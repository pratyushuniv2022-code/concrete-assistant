# config_env.py
import os
from pathlib import Path

# ---------------- Default local values ----------------
DEFAULT_QDRANT_HOST = "127.0.0.1"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_CORE_COLL = "core_local_384"
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"   # 384-d sentence-transformers
DEFAULT_CORE_PDF_DIR = Path("core_pdfs")
DEFAULT_BATCH_SIZE = 128
DEFAULT_RETRY_ATTEMPTS = 6
DEFAULT_RETRY_DELAY = 1.5

# ---------------- Qdrant Cloud values ----------------
# Replace these with your actual Qdrant Cloud URL & API Key
CLOUD_QDRANT_URL = "https://d9da8062-7ac5-46ca-9a98-cdb6561e520c.europe-west3-0.gcp.cloud.qdrant.io:6333"
CLOUD_QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.1PSdadgsA0bI7aN_mBcD5AFk4zksd5ioaS-GeaTLfLA"

# ---------------- Read from environment (fallback to defaults) ----------------
QDRANT_HOST = os.environ.get("QDRANT_HOST", DEFAULT_QDRANT_HOST)
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", DEFAULT_QDRANT_PORT))
QDRANT_URL = os.environ.get("QDRANT_URL", CLOUD_QDRANT_URL)        # Cloud-ready
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", CLOUD_QDRANT_API_KEY)

CORE_COLL = os.environ.get("CORE_COLL", DEFAULT_CORE_COLL)
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", DEFAULT_LOCAL_MODEL)

CORE_PDF_DIR = Path(os.environ.get("CORE_PDF_DIR", DEFAULT_CORE_PDF_DIR))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH_SIZE))
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", DEFAULT_RETRY_ATTEMPTS))
RETRY_DELAY = float(os.environ.get("RETRY_DELAY", DEFAULT_RETRY_DELAY))

# Optional: serve PDFs from Cloud Storage
PDF_BUCKET = os.environ.get("PDF_BUCKET")   # if provided, use to build public URLs
PDF_URLS_JSON = os.environ.get("PDF_URLS_JSON")  # optional JSON mapping of name->url
