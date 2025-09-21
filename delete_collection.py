from qdrant_client import QdrantClient

QDRANT_HOST = "127.0.0.1"
QDRANT_PORT = 6333
COLLECTION_NAME = "core_local_384"

# Connect to Qdrant
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

# Delete the collection
qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
