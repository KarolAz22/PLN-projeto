from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch


# =========================================================
# 🔍 1. VERIFICAR VOCABULÁRIO
# =========================================================
def verificar_vocabulario(modelo, lista_palavras):
    tokenizer = AutoTokenizer.from_pretrained(modelo)

    print(f"\n🔍 Analisando modelo: {modelo}")
    print(f"📚 Tamanho do vocabulário: {len(tokenizer.vocab)}")
    print("-" * 50)

    resultados = {}
    unk_token_id = tokenizer.unk_token_id

    for palavra in lista_palavras:
        ids = tokenizer.encode(palavra, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)

        chars_conhecidos = sum(len(t.replace('##', '')) for t in tokens if t != '[UNK]')
        chars_totais = len(palavra.replace(' ', ''))
        cobertura = (chars_conhecidos / chars_totais) * 100 if chars_totais > 0 else 0

        tem_unk = unk_token_id in ids if unk_token_id is not None else False

        resultados[palavra] = {
            'tokens': tokens,
            'ids': ids,
            'tem_unk': tem_unk,
            'cobertura': cobertura
        }

        status = "❌" if tem_unk else "✅"
        print(f"{status} '{palavra}'")
        print(f"   → Tokens: {tokens}")
        print(f"   → IDs: {ids}")
        print(f"   → Cobertura: {cobertura:.1f}%\n")

    return resultados


# =========================================================
# 📊 2. TESTE DE EMBEDDINGS (CORRIGIDO)
# =========================================================
def testar_embeddings(modelo_name):
    print(f"\n\n📊 TESTE DE EMBEDDINGS")
    print(f"Modelo: {modelo_name}")
    print("=" * 60)

    model = SentenceTransformer(modelo_name)

    textos = [
        "A Amenorreia é um sinal marcante da menopausa.",
        "A Atrofia vaginal pode causar desconforto íntimo.",
        "A Osteoporose aumenta o risco de fraturas.",
        "A Labilidade emocional causa mudanças de humor frequentes.",
        "Ultimamente tô com dor na relação, pode ser Dispareunia.",
        "Do nada me dá um calorão, esses Fogachos são terríveis.",
        "Tô acordando encharcada à noite",
        "Uma hora tô bem, outra tô irritada",
        "Gatos são animais domésticos"
    ]

    print("\n1️⃣ Gerando embeddings...")
    embeddings = model.encode(textos, convert_to_tensor=True)

    print(f"✅ Embeddings gerados!")
    print(f"Forma: {embeddings.shape}")

    print("\n2️⃣ Comparando com texto irrelevante (gatos)\n")

    # ✅ corrigido: índice 8 = gatos
    sim_gatos = util.pytorch_cos_sim(embeddings[8], embeddings[:8])

    for i, texto in enumerate(textos[:8]):
        score = sim_gatos[0][i].item()
        print(f"📌 '{texto[:40]}...'")
        print(f"   Similaridade com 'gatos': {score:.4f}\n")

    print("3️⃣ Similaridade entre textos médicos:\n")
    sim_matrix = util.pytorch_cos_sim(embeddings[:3], embeddings[:3])

    for i in range(3):
        for j in range(i + 1, 3):
            print(f"Texto {i+1} ↔ Texto {j+1}: {sim_matrix[i][j]:.4f}")

    return model


# =========================================================
# 🧪 3. TESTE TÉCNICO VS POPULAR + UNK
# =========================================================
def analisar_par(model, tokenizer, texto1, texto2):
    emb = model.encode([texto1, texto2], convert_to_tensor=True)
    similaridade = util.pytorch_cos_sim(emb[0], emb[1]).item()

    tokens1 = tokenizer.tokenize(texto1)
    tokens2 = tokenizer.tokenize(texto2)

    ids1 = tokenizer.encode(texto1, add_special_tokens=False)
    ids2 = tokenizer.encode(texto2, add_special_tokens=False)

    unk_id = tokenizer.unk_token_id
    tem_unk1 = unk_id in ids1 if unk_id is not None else False
    tem_unk2 = unk_id in ids2 if unk_id is not None else False

    print("=" * 60)
    print("🧪 Comparação:\n")
    print(f"📘 Técnico: {texto1}")
    print(f"💬 Popular: {texto2}\n")

    print(f"🔗 Similaridade: {similaridade:.4f}\n")

    print("🔍 Tokens técnico:")
    print(tokens1)
    print(f"UNK: {'❌' if tem_unk1 else '✅'}\n")

    print("🔍 Tokens popular:")
    print(tokens2)
    print(f"UNK: {'❌' if tem_unk2 else '✅'}\n")


def testar_tecnico_vs_popular():
    modelo_embeddings = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    modelo_tokenizer = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    print("\n📦 Carregando modelos...\n")
    model = SentenceTransformer(modelo_embeddings)
    tokenizer = AutoTokenizer.from_pretrained(modelo_tokenizer)

    pares = [
        ("Fogachos são comuns na menopausa", "Do nada me dá um calorão"),
        ("Labilidade emocional é frequente", "Uma hora tô bem, outra tô irritada"),
        ("Dispareunia pode ocorrer", "Tô com dor na relação"),
        ("Amenorreia é a ausência de menstruação", "Já faz meses que não menstruo"),
        ("Sudorese noturna é um sintoma comum", "Tô acordando encharcada à noite")
    ]

    print("🚀 TESTE: Técnico vs Popular + UNK")
    print("=" * 60)

    for tecnico, popular in pares:
        analisar_par(model, tokenizer, tecnico, popular)


# =========================================================
# ▶️ EXECUÇÃO
# =========================================================
if __name__ == "__main__":
    print("🚀 INICIANDO TESTES")
    print("=" * 60)

    # 1. Vocabulário
    modelo_vocab = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    termos = [
        "estrogênio",
        "perimenopausa",
        "fogacho",
        "labilidade emocional",
        "síndrome geniturinária da menopausa"
    ]
    verificar_vocabulario(modelo_vocab, termos)

    # 2. Embeddings
    modelo_emb = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = testar_embeddings(modelo_emb)

    # 3. Técnico vs popular
    testar_tecnico_vs_popular()

    print("\n✅ Tudo finalizado!")