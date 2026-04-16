import sys
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os

load_dotenv()

# Adiciona o caminho do encoder
sys.path.insert(0, str(Path(__file__).parent))
from encoder.biobertpt_encoder_v2 import BioBERTptEncoderV2

# ==== CONFIGURAÇÕES ====
QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"')
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip().strip('"')
COLLECTION_NAME = "Tide"

print("=" * 80)
print("🔍 TESTE DE BUSCA SEMÂNTICA NO QDRANT")
print("=" * 80)

# ==== PASSO 1: Conectar ao Qdrant ====
print("\n🔹 Conectando ao Qdrant Cloud...")
try:
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    print("✅ Conexão OK!")
except Exception as e:
    print(f"❌ Erro ao conectar: {e}")
    sys.exit(1)

# ==== PASSO 2: Informações da coleção ====
print(f"\n🔹 Informações da coleção '{COLLECTION_NAME}':")
try:
    collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
    print(f"  ✅ Total de vetores: {collection_info.points_count}")
    print(f"  ✅ Vetores indexados: {collection_info.indexed_vectors_count}")
    print(f"  ✅ Dimensão: {collection_info.config.params.vectors.size}")
except Exception as e:
    print(f"❌ Erro ao obter informações: {e}")
    sys.exit(1)

# ==== PASSO 3: Listar alguns exemplos ====
print(f"\n🔹 Primeiros 5 documentos indexados:")
try:
    points = qdrant.scroll(collection_name=COLLECTION_NAME, limit=5)[0]
    for i, point in enumerate(points, 1):
        payload = point.payload
        print(f"\n  [{i}] ID: {point.id}")
        print(f"      Fonte: {payload.get('fonte', 'N/A')}")
        print(f"      Texto: {payload.get('texto', '')[:100]}...")
except Exception as e:
    print(f"❌ Erro ao listar documentos: {e}")

# ==== PASSO 4: Testar busca semântica ====
print(f"\n{'='*80}")
print("🔹 Teste de Busca Semântica:")
print(f"{'='*80}")

# Carrega o encoder
print("\n📥 Carregando encoder BioBeRT...")
try:
    encoder = BioBERTptEncoderV2(model_name="pucpr/biobertpt-all", pooling="mean")
    print("✅ Encoder carregado!")
except Exception as e:
    print(f"❌ Erro ao carregar encoder: {e}")
    sys.exit(1)

# Realiza busca
query = "proteínas e genes"
print(f"\n🔍 Buscando por: '{query}'")
try:
    # Gera embedding da query
    query_embedding = encoder.encode(query)
    
    # Busca no Qdrant
    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=5
    )
    
    print(f"\n✅ Encontrados {len(search_results)} resultados:\n")
    for i, result in enumerate(search_results, 1):
        payload = result.payload
        score = result.score
        print(f"  [{i}] Similaridade: {score:.4f}")
        print(f"      Fonte: {payload.get('fonte', 'N/A')}")
        print(f"      Texto: {payload.get('texto', '')[:150]}...")
        print()
    
except Exception as e:
    print(f"❌ Erro na busca: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
print("✅ Teste concluído!")
print("=" * 80)
