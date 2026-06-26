import os
from qdrant_client import AsyncQdrantClient

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

qdrant_client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)