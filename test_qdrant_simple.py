from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"')
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip().strip('"')

print(f"URL: {QDRANT_URL}")
print(f"API Key: {QDRANT_API_KEY[:30]}...")

try:
    print("\n🔹 Conectando ao Qdrant Cloud...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
    info = client.get_collections()
    print("✅ Conexão OK!")
    print(f"Coleções existentes: {len(info.collections)}")
    for col in info.collections:
        print(f"  - {col.name}: {col.vectors_count} vetores")
except Exception as e:
    print(f"❌ Erro ao conectar: {type(e).__name__}")
    print(f"Detalhes: {str(e)}")
