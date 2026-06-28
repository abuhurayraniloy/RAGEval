import os
from qdrant_client import AsyncQdrantClient

qdrant_client = qdrant_client = AsyncQdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
