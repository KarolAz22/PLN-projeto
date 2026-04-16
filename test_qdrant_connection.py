import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print(f"URL: {QDRANT_URL}")
print(f"API Key: {QDRANT_API_KEY[:20] if QDRANT_API_KEY else 'NONE'}...")

try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
    collections = qdrant.get_collections()
    print("\n✅ Conexão bem-sucedida!")
    print(f"Coleções existentes: {[c.name for c in collections.collections]}")
except Exception as e:
    print(f"\n❌ Erro ao conectar: {type(e).__name__}: {e}")
