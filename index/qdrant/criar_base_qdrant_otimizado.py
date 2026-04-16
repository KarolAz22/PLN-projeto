import json
import sys
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import os
import numpy as np
import time

load_dotenv()

# Adiciona o caminho do encoder
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from encoder.biobertpt_encoder_v2 import BioBERTptEncoderV2

# ==== CONFIGURAÇÕES ====
CHUNKS_PATH = Path("index/files/doc_chunks.jsonl")
COLLECTION_NAME = "Tide"
BATCH_SIZE = 16  # Reduzido ainda mais para evitar problemas de memória
EMBEDDING_DIM = 768  # BioBeRT sempre 768

# Carrega variáveis de ambiente
QDRANT_URL = os.getenv("QDRANT_URL", "").strip().strip('"')
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip().strip('"')

if not QDRANT_URL or not QDRANT_API_KEY:
    print("❌ QDRANT_URL ou QDRANT_API_KEY não configurados!")
    sys.exit(1)

print(f"✅ Configurações carregadas")
print(f"  URL: {QDRANT_URL[:50]}...")
print(f"  API Key: {QDRANT_API_KEY[:30]}...\n")

# ==== PASSO 1: Conectar ao Qdrant Cloud ====
print("🔹 Conectando ao Qdrant Cloud...")
try:
    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60,
    )
    # Testa a conexão
    info = qdrant.get_collections()
    print(f"✅ Conexão OK! (Coleções existentes: {len(info.collections)})\n")
except Exception as e:
    print(f"❌ Erro ao conectar: {e}")
    sys.exit(1)

# ==== PASSO 2: Criar ou verificar a coleção (LOGO APÓS CONECTAR) ====
print(f"🔹 Criando/Verificando a coleção '{COLLECTION_NAME}'...")
try:
    # Tenta deletar coleção antiga se existir
    try:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        print(f"  ✅ Coleção antiga deletada")
        time.sleep(2)
    except:
        pass
    
    # Cria nova coleção IMEDIATAMENTE
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM,
            distance=models.Distance.COSINE
        )
    )
    print(f"✅ Coleção '{COLLECTION_NAME}' criada com sucesso!\n")
except Exception as e:
    print(f"❌ Erro ao criar coleção: {e}")
    sys.exit(1)

# ==== PASSO 3: Carregar encoder ====
print("🔹 Carregando encoder BioBERTpt V2...")
try:
    encoder = BioBERTptEncoderV2(model_name="pucpr/biobertpt-all", pooling="mean")
    print(f"✅ Encoder carregado!\n")
except Exception as e:
    print(f"❌ Erro ao carregar encoder: {e}")
    sys.exit(1)

# ==== PASSO 4: Contar total de chunks ====
print("🔹 Contando chunks...")
total_chunks = 0
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    total_chunks = sum(1 for _ in f)
print(f"✅ {total_chunks} chunks encontrados\n")

# ==== PASSO 5: Processar e inserir chunks em batches ====
print("🔹 Processando e inserindo chunks...")
chunk_id = 0
batch_texts = []
batch_chunks = []
total_inserted = 0

try:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_chunks, desc="Progress"):
            chunk = json.loads(line)
            batch_texts.append(chunk.get("chunk_text", ""))
            batch_chunks.append(chunk)
            
            # Quando atingir o tamanho do batch, processa e insere
            if len(batch_texts) >= BATCH_SIZE:
                try:
                    # Gera embeddings
                    embeddings = encoder.encode_batch(batch_texts)
                    
                    # Prepara points
                    points = []
                    for chunk, emb in zip(batch_chunks, embeddings):
                        fonte = (
                            chunk.get("source")
                            or chunk.get("metadata", {}).get("source")
                            or "unknown"
                        )
                        
                        payload = {
                            "id_original": chunk.get("original_id", ""),
                            "indice_de_blocos": chunk.get("chunk_index", 0),
                            "texto": chunk.get("chunk_text", ""),
                            "fonte": fonte,
                        }
                        
                        points.append(
                            models.PointStruct(
                                id=chunk_id,
                                vector=emb.tolist(),
                                payload=payload
                            )
                        )
                        chunk_id += 1
                    
                    # Insere no Qdrant
                    qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    total_inserted += len(points)
                    
                except Exception as e:
                    print(f"\n❌ Erro ao processar batch: {e}")
                    raise
                
                # Reset batch
                batch_texts = []
                batch_chunks = []
        
        # Processa chunks restantes
        if batch_texts:
            try:
                embeddings = encoder.encode_batch(batch_texts)
                points = []
                for chunk, emb in zip(batch_chunks, embeddings):
                    fonte = (
                        chunk.get("source")
                        or chunk.get("metadata", {}).get("source")
                        or "unknown"
                    )
                    
                    payload = {
                        "id_original": chunk.get("original_id", ""),
                        "indice_de_blocos": chunk.get("chunk_index", 0),
                        "texto": chunk.get("chunk_text", ""),
                        "fonte": fonte,
                    }
                    
                    points.append(
                        models.PointStruct(
                            id=chunk_id,
                            vector=emb.tolist(),
                            payload=payload
                        )
                    )
                    chunk_id += 1
                
                qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                total_inserted += len(points)
            except Exception as e:
                print(f"\n❌ Erro ao processar últimos chunks: {e}")
                raise

except Exception as e:
    print(f"\n❌ Erro geral no processamento: {e}")
    sys.exit(1)

# ==== PASSO 6: Verificação final ====
print(f"\n✅ Processamento concluído!")
print(f"   Total de chunks inseridos: {total_inserted}")

try:
    collection_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
    print(f"\n📊 Coleção '{COLLECTION_NAME}' - Informações finais:")
    print(f"   ✅ Total de vetores: {collection_info.points_count}")
    print(f"   ✅ Tamanho do índice: {collection_info.indexed_vectors_count} vetores indexados")
except Exception as e:
    print(f"⚠️  Erro ao obter informações finais: {e}")

print(f"\n🎉 Indexação com BioBeRT concluída com sucesso!")
