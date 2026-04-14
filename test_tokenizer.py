from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

def verificar_vocabulario(modelo, lista_palavras):
    """
    Verifica se palavras estão no vocabulário do modelo.
    
    Args:
        modelo: Nome do modelo no HuggingFace
        lista_palavras: Lista de palavras/frases para testar
    """
    tokenizer = AutoTokenizer.from_pretrained(modelo)
    
    print(f"\n🔍 Analisando modelo: {modelo}")
    print(f"📚 Tamanho do vocabulário: {len(tokenizer.vocab)}")
    print("-" * 50)
    
    resultados = {}
    for palavra in lista_palavras:
        ids = tokenizer.encode(palavra, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        
        # Calcula % do texto que é conhecido
        chars_conhecidos = sum(len(t.replace('##', '')) for t in tokens if t != '[UNK]')
        chars_totais = len(palavra.replace(' ', ''))
        cobertura = (chars_conhecidos / chars_totais) * 100 if chars_totais > 0 else 0
        
        resultados[palavra] = {
            'tokens': tokens,
            'ids': ids,
            'tem_unk': 100 in ids,
            'cobertura': cobertura
        }
        
        status = "❌" if 100 in ids else "✅"
        print(f"{status} '{palavra}'")
        print(f"   → Tokens: {tokens}")
        print(f"   → IDs: {ids}")
        print(f"   → Cobertura: {cobertura:.1f}%\n")
    
    return resultados


def testar_embeddings(modelo_name):
    """
    Testa a geração de embeddings e similaridade entre textos.
    
    Args:
        modelo_name: Nome do modelo (SentenceTransformer)
    """
    print(f"\n\n📊 TESTE DE EMBEDDINGS")
    print(f"Modelo: {modelo_name}")
    print("=" * 60)
    
    try:
        # Carregar modelo
        model = SentenceTransformer(modelo_name)
        
        # Textos de teste
        textos = [
            "Menopausa é a cessação da menstruação",
            "A perimenopausa precede a menopausa",
            "Fogachos são sintomas da menopausa",
            "O tratamento hormonal alivia sintomas",
            "Gatos são animais domésticos"
        ]
        
        # Gerar embeddings
        print("\n1️⃣ Gerando embeddings...")
        embeddings = model.encode(textos, convert_to_tensor=True)
        print(f"✅ Embeddings gerados com sucesso!")
        print(f"   → Forma: {embeddings.shape}")
        print(f"   → Dimensionalidade: {embeddings.shape[1]}")
        
        # Calcular similaridade
        print("\n2️⃣ Calculando similaridades...")
        print("   (Entre textos sobre menopausa vs texto sobre gatos)\n")
        
        # Similaridade do texto sobre gatos com os outros
        sim_gatos = util.pytorch_cos_sim(embeddings[4], embeddings[:4])
        
        for i, texto in enumerate(textos[:4]):
            score = sim_gatos[0][i].item()
            print(f"   📌 '{texto[:40]}...'")
            print(f"      vs 'Gatos são animais domésticos': {score:.4f}\n")
        
        # Similaridade entre textos de menopausa
        print("3️⃣ Similaridade entre textos sobre menopausa:\n")
        sim_matrix = util.pytorch_cos_sim(embeddings[:3], embeddings[:3])
        
        for i in range(3):
            for j in range(i+1, 3):
                score = sim_matrix[i][j].item()
                print(f"   🔗 Texto {i+1} ↔ Texto {j+1}: {score:.4f}")
        
        return model, embeddings
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("   Sugerindo fallback...")
        return None, None


# EXECUTANDO TESTES
if __name__ == "__main__":
    import sys
    
    # Modo rápido vs completo
    modo_rapido = False  # Mude para False se quer teste completo
    
    print("🚀 INICIANDO TESTE DE ENCODER/TOKENIZER")
    print(f"Modo: {'⚡ RÁPIDO' if modo_rapido else '🔬 COMPLETO'}")
    print("=" * 60)
    
    if modo_rapido:
        # MODO RÁPIDO: Apenas verificar tokens (sem baixar embeddings)
        print("\n📦 Carregando apenas tokenizer (mais rápido)...\n")
        modelo_teste = "efederici/sentence-bert-base"
        termos = ["menopausa", "estrogênio", "fogacho"]
        resultados = verificar_vocabulario(modelo_teste, termos)
        
    else:
        # MODO COMPLETO: Testes de tokens + embeddings
        print("\n📦 Carregando modelos (pode demorar)...\n")
        
        # Teste 1: Vocabulário
        modelo_teste = "efederici/sentence-bert-base"
        termos_menopausa = [
            "estrogênio", 
            "perimenopausa", 
            "menopausa",
            "fogacho",
            "síndrome geniturinária da menopausa"
        ]
        resultados = verificar_vocabulario(modelo_teste, termos_menopausa)
        
        # Teste 2: Embeddings (usando sentence-transformers)
        print("\n" + "=" * 60)
        modelo_embeddings = "distiluse-base-multilingual-cased-v2"
        model, embeddings = testar_embeddings(modelo_embeddings)
    
    print("\n" + "=" * 60)
    print("✅ Testes concluídos!")
    print("\n💡 Dica: Para usar modo completo, mude 'modo_rapido = False' no código")