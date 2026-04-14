📊 ÍNDICE DE RESULTADOS DOS TESTES
================================================================================

Este arquivo lista todos os scripts de comparação/teste e seus arquivos de resultados.

1. COMPARE_MODELS.PY
   📂 Localização: /PLN-projeto/
   📄 Arquivo de resultados: compare_models_results.txt
   🔬 Descrição: Compara BioBERTpt vs Paraphrase-MiniLM-L12-v2
   📍 Testes:
      • Tempo de carregamento
      • Similaridade com palavras individuais (3 pares)
      • Similaridade com frases completas (2 pares)
      • Métrica de discriminação semântica
      • Conclusão: qual modelo é melhor

2. TEST_COMPARE.PY
   📂 Localização: /PLN-projeto/encoder/
   📄 Arquivo de resultados: test_compare_results.txt
   🔬 Descrição: Compara paraphrase-multilingual-MiniLM v1 vs v2
   📍 Testes:
      • Tempo de carregamento de ambas as versões
      • Similaridade em pares relacionados (5 pares)
      • Similaridade em pares não relacionados (5 pares)
      • Métrica de discriminação semântica
      • Teste de performance (tempo de encode)
      • Recomendação final

3. COMPARE_TOKENIZERS.PY
   📂 Localização: /PLN-projeto/
   📄 Arquivo de resultados: compare_tokenizers_results.txt
   🔬 Descrição: Compara test_tokenizer.py vs test_tokenizer_2.py
   📍 Testes:
      • Análise de vocabulário (cobertura de termos)
      • BioBERTpt vs Paraphrase-MiniLM
      • Comparação de embeddings
      • Resumo de resultados técnicos

================================================================================
EXECUTAR OS TESTES:

Bash no diretório /PLN-projeto:

# Comparar modelos principales
python3 compare_models.py

# Comparar versões do paraphrase-multilingual (no encoder/)
cd encoder && python3 test_compare.py && cd ..

# Comparar testes de tokenizer
python3 compare_tokenizers.py

================================================================================
RESULTADOS GERADOS:

✅ compare_models_results.txt
✅ encoder/test_compare_results.txt
✅ compare_tokenizers_results.txt

================================================================================
Última atualização: 14/04/2026
