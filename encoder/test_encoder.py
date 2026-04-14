import sys
import time
import numpy as np
from biobertpt_encoder import BioBERTptEncoder

print("=" * 70)
print("🚀 TESTE COMPLETO DO ENCODER BioBERTpt")
print("=" * 70)

# 1. Inicializa o encoder
print("\n1️⃣ Inicializando encoder...")
start_time = time.time()
encoder = BioBERTptEncoder()
init_time = time.time() - start_time
print(f"   ✅ Inicializado em {init_time:.2f} segundos")
print(f"   📱 Dispositivo: {encoder.device}")
print(f"   📚 Vocabulário: {len(encoder.tokenizer)} tokens")

# 2. Teste com termos clínicos individuais
print("\n2️⃣ Testando com termos clínicos...")
termos_clinicos = [
    "menopausa",
    "perimenopausa",
    "fogacho",
    "estrogênio",
    "síndrome geniturinária da menopausa",
    "labilidade emocional",
    "atrofia vaginal",
    "osteoporose"
]

for termo in termos_clinicos:
    emb = encoder.encode(termo)
    print(f"   ✓ '{termo}':")
    print(f"      → Shape: {emb.shape}")
    print(f"      → Norma: {np.linalg.norm(emb):.4f}")
    print(f"      → Primeiros 3 valores: [{emb[0]:.4f}, {emb[1]:.4f}, {emb[2]:.4f}]")

# 3. Teste de similaridade semântica
print("\n3️⃣ Testando similaridade semântica...")
textos_teste = [
    "A menopausa é a cessação permanente da menstruação",
    "Fogachos são episódios de calor súbito com sudorese",
    "A terapia hormonal repõe estrogênio e progesterona",
    "Gatos são animais domésticos muito populares"
]

# Gera embeddings em batch
embeddings = encoder.encode_batch(textos_teste)

print("\n   Matriz de similaridade:")
print("   " + "-" * 60)
for i in range(len(textos_teste)):
    linha = f"   Texto {i+1}: "
    for j in range(len(textos_teste)):
        sim = encoder.similaridade_cosseno(embeddings[i], embeddings[j])
        linha += f"{sim:.3f}  "
    print(linha)
print("   " + "-" * 60)

# 4. Teste de performance (tempo de processamento)
print("\n4️⃣ Testando performance...")
texto_medio = "A paciente apresenta fogachos frequentes e sudorese noturna durante a perimenopausa"

# Executa 20 vezes e calcula média
num_execucoes = 20
tempos = []

for i in range(num_execucoes):
    start = time.time()
    emb = encoder.encode(texto_medio)
    tempos.append(time.time() - start)

tempo_medio = sum(tempos) / len(tempos)
tempo_min = min(tempos)
tempo_max = max(tempos)

print(f"   Texto: '{texto_medio[:60]}...'")
print(f"   Execuções: {num_execucoes}")
print(f"   Tempo médio: {tempo_medio*1000:.2f}ms")
print(f"   Tempo mínimo: {tempo_min*1000:.2f}ms")
print(f"   Tempo máximo: {tempo_max*1000:.2f}ms")

# 5. Teste de batch processing
print("\n5️⃣ Testando processamento em lote (batch)...")
tamanhos_batch = [1, 10, 50, 100]
texto_base = "Texto sobre menopausa e seus sintomas"

for tamanho in tamanhos_batch:
    textos = [f"{texto_base} {i}" for i in range(tamanho)]
    
    start = time.time()
    embeddings_batch = encoder.encode_batch(textos)
    tempo_batch = time.time() - start
    
    print(f"   Batch size {tamanho:3d}: {tempo_batch:.3f}s → {tempo_batch/tamanho*1000:.2f}ms por texto")

# 6. Teste com termos relacionados (coerência semântica)
print("\n6️⃣ Testando coerência semântica...")
pares_relacionados = [
    ("menopausa", "climatério"),
    ("fogacho", "calorão"),
    ("estrogênio", "hormônio"),
    ("dispareunia", "dor na relação"),
]

pares_nao_relacionados = [
    ("menopausa", "computador"),
    ("fogacho", "carro"),
    ("estrogênio", "cachorro"),
]

print("\n   Pares relacionados (similaridade deve ser ALTA):")
for palavra1, palavra2 in pares_relacionados:
    emb1 = encoder.encode(palavra1)
    emb2 = encoder.encode(palavra2)
    sim = encoder.similaridade_cosseno(emb1, emb2)
    print(f"   ✓ '{palavra1}' ↔ '{palavra2}': {sim:.4f}")

print("\n   Pares não relacionados (similaridade deve ser BAIXA):")
for palavra1, palavra2 in pares_nao_relacionados:
    emb1 = encoder.encode(palavra1)
    emb2 = encoder.encode(palavra2)
    sim = encoder.similaridade_cosseno(emb1, emb2)
    print(f"   ✗ '{palavra1}' ↔ '{palavra2}': {sim:.4f}")

# 7. Estatísticas finais
print("\n7️⃣ Estatísticas finais...")
print(f"   Total de textos processados: {len(termos_clinicos) + len(textos_teste) + num_execucoes + sum(tamanhos_batch)}")
print(f"   Dimensionalidade dos embeddings: {embeddings.shape[1]}")
print(f"   Modelo: {encoder.model_name}")

print("\n" + "=" * 70)
print("✅ TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
print("=" * 70)

# Teste adicional: exemplo prático de busca
print("\n📌 EXEMPLO PRÁTICO: Busca de documentos similares")
print("-" * 70)

documentos = [
    "A menopausa ocorre naturalmente entre os 45 e 55 anos",
    "Fogachos afetam cerca de 75% das mulheres na menopausa",
    "A terapia de reposição hormonal é eficaz para sintomas vasomotores",
    "A osteoporose pós-menopausa aumenta risco de fraturas",
    "A síndrome geniturinária causa desconforto e dor durante a relação",
    "Gatos são animais independentes e exigem poucos cuidados"
]

consulta = "calorão e suor durante a noite"

print(f"\nConsulta: '{consulta}'\n")
print("Documentos mais similares:")

# Gera embedding da consulta
emb_consulta = encoder.encode(consulta)

# Calcula similaridade com cada documento
resultados = []
for i, doc in enumerate(documentos):
    emb_doc = encoder.encode(doc)
    sim = encoder.similaridade_cosseno(emb_consulta, emb_doc)
    resultados.append((i, sim, doc))

# Ordena por similaridade
resultados.sort(key=lambda x: x[1], reverse=True)

for i, (idx, sim, doc) in enumerate(resultados[:3], 1):
    print(f"\n{i}. Similaridade: {sim:.4f}")
    print(f"   {doc}")

print("\n" + "=" * 70)