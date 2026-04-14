import sys
import time
sys.path.insert(0, '/home/paloma/Documentos/20261/pln/PLN-projeto')

from encoder.paraphrase_multilingual_encoder_v2 import ParaphraseMultilingualEncoderV2
from encoder.biobertpt_encoder_v2 import BioBERTptEncoderV2
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np

def verificar_vocabulario(modelo, lista_palavras):
    """
    Verifica se palavras estão no vocabulário do modelo.
    """
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

def testar_embeddings_encoder(encoder, nome_modelo, textos):
    """
    Testa embeddings usando o encoder customizado.
    """
    print(f"\n📊 TESTE DE EMBEDDINGS com {nome_modelo}")
    print("=" * 60)

    print("\n1️⃣ Gerando embeddings...")
    embeddings = []
    for texto in textos:
        emb = encoder.encode(texto)
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    print(f"✅ Embeddings gerados!")
    print(f"Forma: {embeddings.shape}")

    print("\n2️⃣ Comparando com texto irrelevante (último)\n")

    texto_irrelevante = textos[-1]
    emb_irrelevante = embeddings[-1]

    for i, texto in enumerate(textos[:-1]):
        sim = encoder.similaridade_cosseno(embeddings[i], emb_irrelevante)
        print(f"📌 '{texto[:40]}...'")
        print(f"   Similaridade com '{texto_irrelevante[:20]}...': {sim:.4f}\n")

    print("3️⃣ Similaridade entre textos relacionados:\n")
    for i in range(len(textos)-1):
        for j in range(i+1, len(textos)-1):
            sim = encoder.similaridade_cosseno(embeddings[i], embeddings[j])
            print(f"Texto {i+1} ↔ Texto {j+1}: {sim:.4f}")

    return embeddings

def comparar_modelos_vocab(modelos, termos_teste1, termos_teste2):
    """
    Compara vocabulário entre modelos para os termos dos testes.
    """
    print("=" * 70)
    print("🔬 COMPARAÇÃO DE VOCABULÁRIO")
    print("=" * 70)

    resultados = {}

    for modelo in modelos:
        print(f"\n🧪 Teste com {modelo}")
        print("-" * 30)

        # Termos do test_tokenizer
        print("📋 Termos do test_tokenizer.py:")
        res1 = verificar_vocabulario(modelo, termos_teste1)

        # Termos do test_tokenizer_2.py
        print("📋 Termos do test_tokenizer_2.py:")
        res2 = verificar_vocabulario(modelo, termos_teste2)

        resultados[modelo] = {'teste1': res1, 'teste2': res2}

    # Comparação final
    print("\n" + "=" * 70)
    print("📊 RESUMO DA COMPARAÇÃO DE VOCABULÁRIO")
    print("=" * 70)

    for termo in set(termos_teste1 + termos_teste2):
        print(f"\n🔍 '{termo}':")
        for modelo in modelos:
            if termo in resultados[modelo]['teste1'] or termo in resultados[modelo]['teste2']:
                res = resultados[modelo]['teste1'].get(termo, resultados[modelo]['teste2'].get(termo))
                status = "❌ UNK" if res['tem_unk'] else "✅ OK"
                print(f"   {modelo}: {status} ({res['cobertura']:.1f}% cobertura)")

    return resultados

def comparar_modelos_embeddings(encoders, textos_teste1, textos_teste2):
    """
    Compara embeddings entre modelos.
    """
    print("\n" + "=" * 70)
    print("🔬 COMPARAÇÃO DE EMBEDDINGS")
    print("=" * 70)

    resultados_emb = {}

    # Teste 1: textos simples
    print("\n📋 Teste com textos do test_tokenizer.py:")
    for nome, encoder in encoders.items():
        emb1 = testar_embeddings_encoder(encoder, nome, textos_teste1)
        resultados_emb[f"{nome}_teste1"] = emb1

    # Teste 2: textos complexos
    print("\n📋 Teste com textos do test_tokenizer_2.py:")
    for nome, encoder in encoders.items():
        emb2 = testar_embeddings_encoder(encoder, nome, textos_teste2)
        resultados_emb[f"{nome}_teste2"] = emb2

    return resultados_emb

# Termos dos testes
termos_teste1 = [
    "estrogênio",
    "perimenopausa",
    "menopausa",
    "fogacho",
    "síndrome geniturinária da menopausa"
]

termos_teste2 = [
    "estrogênio",
    "perimenopausa",
    "fogacho",
    "labilidade emocional",
    "síndrome geniturinária da menopausa"
]

# Textos dos testes
textos_teste1 = [
    "Menopausa é a cessação da menstruação",
    "A perimenopausa precede a menopausa",
    "Fogachos são sintomas da menopausa",
    "O tratamento hormonal alivia sintomas",
    "Gatos são animais domésticos"
]

textos_teste2 = [
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

if __name__ == "__main__":
    print("🚀 INICIANDO COMPARAÇÃO: test_tokenizer vs test_tokenizer_2")
    print("Modelos: BioBERTpt vs Paraphrase-MiniLM")
    print("=" * 70)

    # Carregar encoders
    print("\n📦 Carregando encoders...")
    start = time.time()
    biobert_encoder = BioBERTptEncoderV2(model_name="pucpr/biobertpt-all", pooling="mean")
    tempo_bio = time.time() - start
    print(f"✅ BioBERTpt carregado em {tempo_bio:.2f}s")

    start = time.time()
    multilingual_encoder = ParaphraseMultilingualEncoderV2(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", pooling="mean")
    tempo_multi = time.time() - start
    print(f"✅ Paraphrase-MiniLM carregado em {tempo_multi:.2f}s")

    encoders = {
        "BioBERTpt": biobert_encoder,
        "Paraphrase-MiniLM": multilingual_encoder
    }

    # Modelos para vocabulário
    modelos_vocab = ["pucpr/biobertpt-all", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]

    # Comparar vocabulário
    vocab_results = comparar_modelos_vocab(modelos_vocab, termos_teste1, termos_teste2)

    # Comparar embeddings
    emb_results = comparar_modelos_embeddings(encoders, textos_teste1, textos_teste2)

    print("\n" + "=" * 70)
    print("✅ Comparação concluída!")
    print("Resultados salvos em compare_tokenizers_results.txt")

    # Salvar resultados
    with open("compare_tokenizers_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("🔬 RESULTADOS DA COMPARAÇÃO: test_tokenizer vs test_tokenizer_2\n")
        f.write("Modelos: BioBERTpt vs Paraphrase-MiniLM\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Tempo de carregamento BioBERTpt: {tempo_bio:.2f}s\n")
        f.write(f"Tempo de carregamento Paraphrase-MiniLM: {tempo_multi:.2f}s\n\n")

        f.write("RESUMO VOCABULÁRIO:\n")
        for termo in set(termos_teste1 + termos_teste2):
            f.write(f"\n🔍 '{termo}':\n")
            for modelo in modelos_vocab:
                if termo in vocab_results[modelo]['teste1'] or termo in vocab_results[modelo]['teste2']:
                    res = vocab_results[modelo]['teste1'].get(termo, vocab_results[modelo]['teste2'].get(termo))
                    status = "UNK" if res['tem_unk'] else "OK"
                    f.write(f"   {modelo}: {status} ({res['cobertura']:.1f}% cobertura)\n")

        f.write("\n\nEMBEDDINGS: Ver logs acima para detalhes.\n")