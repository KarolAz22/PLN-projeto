import sys
import time
sys.path.insert(0, '/home/paloma/Documentos/20261/pln/PLN-projeto')

from encoder.paraphrase_multilingual_encoder_v2 import ParaphraseMultilingualEncoderV2
from encoder.biobertpt_encoder_v2 import BioBERTptEncoderV2

print("=" * 70)
print("🔬 COMPARAÇÃO: BioBERTpt vs Paraphrase-MiniLM")
print("=" * 70)

# ============================================================
# 1. CARREGAR OS DOIS MODELOS
# ============================================================

print("\n1️⃣ Carregando BioBERTpt...")
start = time.time()
biobert = BioBERTptEncoderV2(
    model_name="pucpr/biobertpt-all",
    pooling="mean"
)
tempo_bio = time.time() - start
print(f"   ✅ Carregado em {tempo_bio:.2f}s")

print("\n2️⃣ Carregando Paraphrase-MiniLM...")
start = time.time()
sentence = ParaphraseMultilingualEncoderV2(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    pooling="mean"
)
tempo_sentence = time.time() - start
print(f"   ✅ Carregado em {tempo_sentence:.2f}s")

# ============================================================
# 2. TESTE COM PALAVRAS INDIVIDUAIS
# ============================================================

print("\n" + "=" * 70)
print("2️⃣ TESTE COM PALAVRAS INDIVIDUAIS")
print("=" * 70)

pares = [
    ("menopausa", "fogacho"),
    ("menopausa", "estrogênio"),
    ("menopausa", "computador"),
]

print(f"\n{'Par':<35} {'BioBERTpt':<15} {'Paraphrase-MiniLM':<15} {'Diferença':<10}")
print("-" * 75)

for p1, p2 in pares:
    # BioBERTpt
    emb_bio1 = biobert.encode(p1)
    emb_bio2 = biobert.encode(p2)
    sim_bio = biobert.similaridade_cosseno(emb_bio1, emb_bio2)
    
    # Sentence-BERT
    emb_sent1 = sentence.encode(p1)
    emb_sent2 = sentence.encode(p2)
    sim_sent = sentence.similaridade_cosseno(emb_sent1, emb_sent2)
    
    diff = sim_bio - sim_sent
    print(f"{p1} ↔ {p2:<20} {sim_bio:.4f}       {sim_sent:.4f}       {diff:+.4f}")

# ============================================================
# 3. TESTE COM FRASES COMPLETAS
# ============================================================

print("\n" + "=" * 70)
print("3️⃣ TESTE COM FRASES COMPLETAS")
print("=" * 70)

frases = {
    "menopausa": "A menopausa é a cessação permanente da menstruação",
    "fogacho": "Fogachos são episódios de calor súbito com sudorese",
    "gato": "Gatos são animais domésticos independentes"
}

# Gera embeddings das frases
emb_bio_frases = {}
emb_sent_frases = {}

for nome, texto in frases.items():
    emb_bio_frases[nome] = biobert.encode(texto)
    emb_sent_frases[nome] = sentence.encode(texto)

# Compara similaridades
print(f"\n{'Comparação':<35} {'BioBERTpt':<15} {'Paraphrase-MiniLM':<15} {'Diferença':<10}")
print("-" * 75)

# menopausa ↔ fogacho (relacionado)
sim_bio_rel = biobert.similaridade_cosseno(emb_bio_frases["menopausa"], emb_bio_frases["fogacho"])
sim_sent_rel = sentence.similaridade_cosseno(emb_sent_frases["menopausa"], emb_sent_frases["fogacho"])
print(f"{'menopausa ↔ fogacho':<35} {sim_bio_rel:.4f}       {sim_sent_rel:.4f}       {sim_bio_rel - sim_sent_rel:+.4f}")

# menopausa ↔ gato (não relacionado)
sim_bio_nrel = biobert.similaridade_cosseno(emb_bio_frases["menopausa"], emb_bio_frases["gato"])
sim_sent_nrel = sentence.similaridade_cosseno(emb_sent_frases["menopausa"], emb_sent_frases["gato"])
print(f"{'menopausa ↔ gato':<35} {sim_bio_nrel:.4f}       {sim_sent_nrel:.4f}       {sim_bio_nrel - sim_sent_nrel:+.4f}")

# ============================================================
# 4. MÉTRICA DE DISCRIMINAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("4️⃣ MÉTRICA DE DISCRIMINAÇÃO (quanto maior, melhor)")
print("=" * 70)

discriminacao_bio = sim_bio_rel - sim_bio_nrel
discriminacao_sent = sim_sent_rel - sim_sent_nrel

print(f"\nBioBERTpt:      {discriminacao_bio:.4f}")
print(f"Paraphrase-MiniLM:  {discriminacao_sent:.4f}")

print("\n" + "=" * 70)
print("5️⃣ RESULTADO FINAL")
print("=" * 70)

if discriminacao_sent > discriminacao_bio:
    print("\n🎯 Paraphrase-MiniLM é MELHOR!")
    print(f"   Discriminação {discriminacao_sent:.4f} > {discriminacao_bio:.4f}")
    print("   → Melhor para distinguir textos relacionados de não relacionados")
elif discriminacao_bio > discriminacao_sent:
    print("\n🎯 BioBERTpt é MELHOR!")
    print(f"   Discriminação {discriminacao_bio:.4f} > {discriminacao_sent:.4f}")
else:
    print("\n📊 Os dois modelos têm desempenho similar")

print("\n" + "=" * 70)
print("✅ Teste concluído!")
print("=" * 70)

# Salvar resultados
with open("compare_models_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("🔬 RESULTADOS DA COMPARAÇÃO: BioBERTpt vs Paraphrase-MiniLM\n")
    f.write("=" * 70 + "\n\n")
    
    f.write(f"Tempo de carregamento BioBERTpt: {tempo_bio:.2f}s\n")
    f.write(f"Tempo de carregamento Sentence-BERT: {tempo_sentence:.2f}s\n\n")
    
    f.write("TESTE COM PALAVRAS INDIVIDUAIS\n")
    f.write(f"{'Par':<35} {'BioBERTpt':<15} {'Paraphrase-MiniLM':<15} {'Diferença':<10}\n")
    f.write("-" * 75 + "\n")
    for p1, p2 in pares:
        emb_bio1 = biobert.encode(p1)
        emb_bio2 = biobert.encode(p2)
        sim_bio = biobert.similaridade_cosseno(emb_bio1, emb_bio2)
        
        emb_sent1 = sentence.encode(p1)
        emb_sent2 = sentence.encode(p2)
        sim_sent = sentence.similaridade_cosseno(emb_sent1, emb_sent2)
        
        diff = sim_bio - sim_sent
        f.write(f"{p1} ↔ {p2:<20} {sim_bio:.4f}       {sim_sent:.4f}       {diff:+.4f}\n")
    
    f.write("\nTESTE COM FRASES COMPLETAS\n")
    f.write(f"{'Comparação':<35} {'BioBERTpt':<15} {'Paraphrase-MiniLM':<15} {'Diferença':<10}\n")
    f.write("-" * 75 + "\n")
    f.write(f"{'menopausa ↔ fogacho':<35} {sim_bio_rel:.4f}       {sim_sent_rel:.4f}       {sim_bio_rel - sim_sent_rel:+.4f}\n")
    f.write(f"{'menopausa ↔ gato':<35} {sim_bio_nrel:.4f}       {sim_sent_nrel:.4f}       {sim_bio_nrel - sim_sent_nrel:+.4f}\n")
    
    f.write("\nMÉTRICA DE DISCRIMINAÇÃO\n")
    f.write(f"BioBERTpt:      {discriminacao_bio:.4f}\n")
    f.write(f"Paraphrase-MiniLM:  {discriminacao_sent:.4f}\n")
    
    f.write("\nRESULTADO FINAL\n")
    if discriminacao_sent > discriminacao_bio:
        f.write("Paraphrase-MiniLM é MELHOR!\n")
        f.write(f"Discriminação {discriminacao_sent:.4f} > {discriminacao_bio:.4f}\n")
    elif discriminacao_bio > discriminacao_sent:
        f.write("BioBERTpt é MELHOR!\n")
        f.write(f"Discriminação {discriminacao_bio:.4f} > {discriminacao_sent:.4f}\n")
    else:
        f.write("Os dois modelos têm desempenho similar\n")

print("\n📁 Resultados salvos em 'compare_models_results.txt'")