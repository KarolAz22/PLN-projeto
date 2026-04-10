from transformers import AutoTokenizer

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

# Usando a função
modelo_teste = "pucpr/biobertpt-all"
termos_menopausa = [
    "estrogênio", 
    "perimenopausa", 
    "menopausa",
    "fogacho",
    "síndrome geniturinária da menopausa"
]

resultados = verificar_vocabulario(modelo_teste, termos_menopausa)