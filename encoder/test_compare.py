"""
Teste comparativo entre CLS Pooling (original) e Mean Pooling (V2)
"""

import time
import numpy as np
from sentence_bert_base_encoder import SentenceBertBaseEncoder
from sentence_bert_base_encoder_v2 import SentenceBertBaseEncoderV2

print("=" * 80)
print("🔬 TESTE COMPARATIVO: CLS Pooling vs MEAN Pooling")
print("=" * 80)

# 1. Inicializa ambos os encoders
print("\n1️⃣ Inicializando encoders...")
print("-" * 40)

print("\n📌 Carregando CLS Pooling (original)...")
start = time.time()
encoder_cls = SentenceBertBaseEncoder()
tempo_cls = time.time() - start
print(f"   ✅ Carregado em {tempo_cls:.2f}s")

print("\n📌 Carregando MEAN Pooling (V2)...")
start = time.time()
encoder_mean = SentenceBertBaseEncoderV2(pooling="mean")
tempo_mean = time.time() - start
print(f"   ✅ Carregado em {tempo_mean:.2f}s")

# 2. Testa com pares de palavras
print("\n" + "=" * 80)
print("2️⃣ TESTE DE DISCRIMINAÇÃO SEMÂNTICA")
print("=" * 80)

pares_relacionados = [
    ("menopausa", "climatério"),
    ("fogacho", "calorão"),
    ("estrogênio", "hormônio"),
    ("dispareunia", "dor na relação"),
    ("osteoporose", "fratura"),
]

pares_nao_relacionados = [
    ("menopausa", "computador"),
    ("fogacho", "carro"),
    ("estrogênio", "cachorro"),
    ("dispareunia", "python"),
    ("osteoporose", "futebol"),
]

print("\n📊 Similaridade para PARES RELACIONADOS (deve ser ALTA):")
print("-" * 80)
print(f"{'Par':<35} {'CLS Pooling':<15} {'Mean Pooling':<15} {'Diferença':<10}")
print("-" * 80)

for palavra1, palavra2 in pares_relacionados:
    emb_cls1 = encoder_cls.encode(palavra1)
    emb_cls2 = encoder_cls.encode(palavra2)
    sim_cls = encoder_cls.similaridade_cosseno(emb_cls1, emb_cls2)
    
    emb_mean1 = encoder_mean.encode(palavra1)
    emb_mean2 = encoder_mean.encode(palavra2)
    sim_mean = encoder_mean.similaridade_cosseno(emb_mean1, emb_mean2)
    
    diff = sim_mean - sim_cls
    print(f"{palavra1} ↔ {palavra2:<20} {sim_cls:.4f}       {sim_mean:.4f}       {diff:+.4f}")

print("\n📊 Similaridade para PARES NÃO RELACIONADOS (deve ser BAIXA):")
print("-" * 80)
print(f"{'Par':<35} {'CLS Pooling':<15} {'Mean Pooling':<15} {'Diferença':<10}")
print("-" * 80)

for palavra1, palavra2 in pares_nao_relacionados:
    emb_cls1 = encoder_cls.encode(palavra1)
    emb_cls2 = encoder_cls.encode(palavra2)
    sim_cls = encoder_cls.similaridade_cosseno(emb_cls1, emb_cls2)
    
    emb_mean1 = encoder_mean.encode(palavra1)
    emb_mean2 = encoder_mean.encode(palavra2)
    sim_mean = encoder_mean.similaridade_cosseno(emb_mean1, emb_mean2)
    
    diff = sim_mean - sim_cls
    print(f"{palavra1} ↔ {palavra2:<20} {sim_cls:.4f}       {sim_mean:.4f}       {diff:+.4f}")

# 3. Métrica de discriminação (quanto maior a diferença, melhor)
print("\n" + "=" * 80)
print("3️⃣ MÉTRICA DE DISCRIMINAÇÃO")
print("=" * 80)

# Calcula média das similaridades
media_rel_cls = np.mean([encoder_cls.similaridade_cosseno(
    encoder_cls.encode(p1), encoder_cls.encode(p2)) for p1, p2 in pares_relacionados])

media_rel_mean = np.mean([encoder_mean.similaridade_cosseno(
    encoder_mean.encode(p1), encoder_mean.encode(p2)) for p1, p2 in pares_relacionados])

media_nao_cls = np.mean([encoder_cls.similaridade_cosseno(
    encoder_cls.encode(p1), encoder_cls.encode(p2)) for p1, p2 in pares_nao_relacionados])

media_nao_mean = np.mean([encoder_mean.similaridade_cosseno(
    encoder_mean.encode(p1), encoder_mean.encode(p2)) for p1, p2 in pares_nao_relacionados])

discriminacao_cls = media_rel_cls - media_nao_cls
discriminacao_mean = media_rel_mean - media_nao_mean

print(f"\n{'Métrica':<30} {'CLS Pooling':<20} {'Mean Pooling':<20}")
print("-" * 70)
print(f"{'Média - Relacionados':<30} {media_rel_cls:.4f}{' ' * 15} {media_rel_mean:.4f}")
print(f"{'Média - Não relacionados':<30} {media_nao_cls:.4f}{' ' * 15} {media_nao_mean:.4f}")
print(f"{'DIFERENÇA (Discriminação)':<30} {discriminacao_cls:.4f}{' ' * 15} {discriminacao_mean:.4f}")
print("-" * 70)

if discriminacao_mean > discriminacao_cls:
    print("\n✅ MEAN POOLING é MELHOR! (maior discriminação)")
elif discriminacao_mean < discriminacao_cls:
    print("\n✅ CLS POOLING é MELHOR! (maior discriminação)")
else:
    print("\n⚠️ Ambos têm desempenho similar")

# 4. Teste de performance
print("\n" + "=" * 80)
print("4️⃣ TESTE DE PERFORMANCE")
print("=" * 80)

texto_teste = "A paciente apresenta fogachos frequentes e sudorese noturna durante a menopausa"
num_execucoes = 20

print(f"\nTexto: '{texto_teste[:60]}...'")
print(f"Execuções: {num_execucoes}\n")

# Teste CLS
tempos_cls = []
for _ in range(num_execucoes):
    start = time.time()
    _ = encoder_cls.encode(texto_teste)
    tempos_cls.append(time.time() - start)

tempo_medio_cls = np.mean(tempos_cls) * 1000
tempo_std_cls = np.std(tempos_cls) * 1000

# Teste MEAN
tempos_mean = []
for _ in range(num_execucoes):
    start = time.time()
    _ = encoder_mean.encode(texto_teste)
    tempos_mean.append(time.time() - start)

tempo_medio_mean = np.mean(tempos_mean) * 1000
tempo_std_mean = np.std(tempos_mean) * 1000

print(f"{'Métrica':<20} {'CLS Pooling':<20} {'Mean Pooling':<20}")
print("-" * 60)
print(f"{'Tempo médio (ms)':<20} {tempo_medio_cls:.2f} ms{' ' * 15} {tempo_medio_mean:.2f} ms")
print(f"{'Desvio padrão (ms)':<20} {tempo_std_cls:.2f} ms{' ' * 15} {tempo_std_mean:.2f} ms")

if tempo_medio_mean < tempo_medio_cls:
    print(f"\n⚡ Mean Pooling é {tempo_medio_cls/tempo_medio_mean:.1f}x mais rápido")
else:
    print(f"\n⚡ CLS Pooling é {tempo_medio_mean/tempo_medio_cls:.1f}x mais rápido")

# 5. Recomendação final
print("\n" + "=" * 80)
print("5️⃣ RECOMENDAÇÃO FINAL")
print("=" * 80)

print("\n📌 Baseado nos testes:")
print(f"   • Discriminação: {'Mean Pooling' if discriminacao_mean > discriminacao_cls else 'CLS Pooling'}")
print(f"   • Velocidade: {'Mean Pooling' if tempo_medio_mean < tempo_medio_cls else 'CLS Pooling'}")

if discriminacao_mean > discriminacao_cls and tempo_medio_mean < tempo_medio_cls:
    print("\n🎯 RECOMENDAÇÃO: Use sentence-bert-baseEncoderV2 (Mean Pooling)")
    print("   → Melhor discriminação semântica E mais rápido!")
elif discriminacao_mean > discriminacao_cls:
    print("\n🎯 RECOMENDAÇÃO: Use sentence-bert-baseEncoderV2 (Mean Pooling)")
    print("   → Melhor discriminação semântica (velocidade similar)")
else:
    print("\n🎯 RECOMENDAÇÃO: Use sentence-bert-baseEncoder original (CLS Pooling)")
    print("   → Melhor para seu caso de uso")

print("\n" + "=" * 80)
print("✅ TESTE COMPARATIVO CONCLUÍDO!")
print("=" * 80)