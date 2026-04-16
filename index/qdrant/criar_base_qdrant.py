import json
import sys
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import numpy as np

# Adiciona o caminho do encoder
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from encoder.biobertpt_encoder_v2 import BioBERTptEncoderV2

# ==== CONFIGURAÇÕES ====
CHUNKS_PATH = Path("index/files/doc_chunks.jsonl")
COLLECTION_NAME = "Tide"

# Substitua pelos seus dados do Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

print("🔹 Carregando encoder BioBERTpt V2...")
encoder = BioBERTptEncoderV2(model_name="pucpr/biobertpt-all", pooling="mean")

print("🔹 Conectando ao Qdrant Cloud...")
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,  # Aumentar timeout para 60 segundos
)

# ==== Primeira passagem: contar chunks e determinar dimensão do embedding ====
print("🔹 Contando chunks...")
total_chunks = 0
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    total_chunks = sum(1 for _ in f)

print(f"✅ {total_chunks} chunks encontrados.")

# Gera um embedding de teste para saber a dimensão
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    first_chunk = json.loads(f.readline())
    test_emb = encoder.encode(first_chunk["chunk_text"])
    embedding_dim = len(test_emb)

print(f"✅ Dimensão do embedding: {embedding_dim}")

# ==== Cria (ou recria) a coleção ====
print(f"🔹 Criando/Verificando a coleção '{COLLECTION_NAME}'...")
import time
try:
    # Tenta deletar a coleção existente com retry
    max_retries = 5
    for attempt in range(max_retries):
        try:
            qdrant.delete_collection(collection_name=COLLECTION_NAME)
            print(f"  ✅ Coleção antiga deletada.")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # Espera progressiva: 5, 10, 15, 20 seg
                print(f"  ⚠️  Tentativa {attempt + 1} de deletar falhou. Aguardando {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  ℹ️  Coleção pode não existir, criando nova...")
                pass
except:
    pass

# Cria a coleção com retry
for attempt in range(max_retries):
    try:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE
            )
        )
        print(f"  ✅ Coleção '{COLLECTION_NAME}' criada com sucesso.")
        break
    except Exception as e:
        if attempt < max_retries - 1:
            wait_time = 5 * (attempt + 1)
            print(f"  ⚠️  Erro ao criar coleção (tentativa {attempt + 1}): {type(e).__name__}")
            print(f"     Aguardando {wait_time}s antes de tentar novamente...")
            time.sleep(wait_time)
        else:
            print(f"  ❌ Erro ao criar coleção após {max_retries} tentativas: {e}")
            raise

print("🔹 Processando e inserindo chunks em batches...")
BATCH_SIZE = 32  # Reduzido para menos memória
chunk_id = 0
points = []

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    batch_texts = []
    batch_chunks = []
    
    for line in tqdm(f, total=total_chunks, desc="Processando"):
        chunk = json.loads(line)
        batch_texts.append(chunk["chunk_text"])
        batch_chunks.append(chunk)
        
        # Quando atingir o tamanho do batch, processa
        if len(batch_texts) >= BATCH_SIZE:
            embeddings = encoder.encode_batch(batch_texts)
            
            for chunk, emb in zip(batch_chunks, embeddings):
                fonte = (
                    chunk.get("source")
                    or chunk.get("metadata", {}).get("source")
                )

                payload = {
                    "id_original": chunk.get("original_id"),
                    "indice_de_blocos": chunk.get("chunk_index"),
                    "texto": chunk.get("chunk_text"),
                    "fonte": fonte,
                }

                points.append(models.PointStruct(id=chunk_id, vector=emb.tolist(), payload=payload))
                chunk_id += 1

                # Envia em lotes para o Qdrant
                if len(points) >= BATCH_SIZE:
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                    points = []
            
            batch_texts = []
            batch_chunks = []
    
    # Processa os últimos chunks
    if batch_texts:
        embeddings = encoder.encode_batch(batch_texts)
        
        for chunk, emb in zip(batch_chunks, embeddings):
            fonte = (
                chunk.get("source")
                or chunk.get("metadata", {}).get("source")
            )

            payload = {
                "id_original": chunk.get("original_id"),
                "indice_de_blocos": chunk.get("chunk_index"),
                "texto": chunk.get("chunk_text"),
                "fonte": fonte,
            }

            points.append(models.PointStruct(id=chunk_id, vector=emb.tolist(), payload=payload))
            chunk_id += 1

# Envia os pontos restantes
if points:
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

print("✅ Indexação concluída com sucesso no Qdrant Cloud!")

# ==== 7. Teste de busca ====
query = "Quais os sintomas da menopausa?"
query_vector = encoder.encode(query).tolist()

print(f"\nRealizando busca no Qdrant Cloud...")
search_results = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=3
)

print("\nResultados mais semelhantes:")
for r in search_results:
    print("-" * 80)
    print(f"Score: {r.score:.4f}")
    print(r.payload["texto"][:500])
    print()
