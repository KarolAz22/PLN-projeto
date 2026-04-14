#!/usr/bin/env python3
"""
Script para visualizar todos os resultados dos testes de comparação
"""

import os
from pathlib import Path

def exibir_resultado(caminho_arquivo, titulo):
    """Exibe o conteúdo de um arquivo de resultado"""
    if os.path.exists(caminho_arquivo):
        print(f"\n{'='*80}")
        print(f"📄 {titulo}")
        print(f"📂 {caminho_arquivo}")
        print(f"{'='*80}\n")
        
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                conteudo = f.read()
                print(conteudo)
        except Exception as e:
            print(f"❌ Erro ao ler arquivo: {e}")
    else:
        print(f"\n⚠️ ARQUIVO NÃO ENCONTRADO: {caminho_arquivo}")
        print("   Execute os testes primeiro para gerar os resultados.\n")

def main():
    print("\n" + "="*80)
    print("🔬 VISUALIZADOR DE RESULTADOS DOS TESTES")
    print("="*80)
    
    # Define os arquivos de resultado
    resultados = [
        ("compare_models_results.txt", "COMPARAÇÃO: BioBERTpt vs Paraphrase-MiniLM"),
        ("encoder/test_compare_results.txt", "COMPARAÇÃO: Paraphrase-MiniLM V1 vs V2"),
        ("compare_tokenizers_results.txt", "COMPARAÇÃO: test_tokenizer vs test_tokenizer_2"),
    ]
    
    # Exibe todos os resultados
    for arquivo, titulo in resultados:
        exibir_resultado(arquivo, titulo)
    
    print("\n" + "="*80)
    print("✅ FIM DA VISUALIZAÇÃO")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
