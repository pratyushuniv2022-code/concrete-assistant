# qdrant_init.py
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

def make_qdrant_client(host, port, api_key=None, check_compatibility=False, retries=3):
    """
    Create a QdrantClient that works for local (host like '127.0.0.1') or
    remote cloud URLs (https://...). If api_key provided, uses it.
    """
    # If host looks like a full URL (starts with http), use url param
    try:
        if host.startswith("http://") or host.startswith("https://"):
            # Qdrant Cloud style: pass url and api_key
            client = QdrantClient(url=host, api_key=api_key, prefer_grpc=False)
        else:
            # local host: use host + port
            client = QdrantClient(host=host, port=port, check_compatibility=check_compatibility)
        # lightweight check
        _ = client.get_collections()
        return client
    except Exception as e:
        # propagate after logging
        raise RuntimeError(f"Could not create Qdrant client for host={host}: {e}")

# Example usage:
# from config_env import QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY
# qdrant = make_qdrant_client(QDRANT_HOST, QDRANT_PORT, api_key=QDRANT_API_KEY)
