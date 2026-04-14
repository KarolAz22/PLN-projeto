import json
import os
import glob
from datetime import datetime

def listar_resultados():
    """Lista todos os arquivos de resultado"""
    arquivos = glob.glob("resultados/comparacao_*.json")
    
    if not arquivos:
        print("Nenhum resultado encontrado em 'resultados/'")
        return
    
    print("📁 Resultados disponíveis:")
    for i, arquivo in enumerate(arquivos, 1):
        data = os.path.getmtime(arquivo)
        data_str = datetime.fromtimestamp(data).strftime('%Y-%m-%d %H:%M:%S')
        print(f"   {i}. {os.path.basename(arquivo)} ({data_str})")
    
    return arquivos

def carregar_resultado(arquivo):
    """Carrega e mostra um resultado específico"""
    with open(arquivo, 'r', encoding='utf-8') as f:
        dados = json.load(f)
    
    print("\n" + "="*80)
    print(f"📊 RESULTADOS: {os.path.basename(arquivo)}")
    print("="*80)
    
    # Ranking
    ranking = sorted(dados.items(), key=lambda x: x[1]['discriminacao'], reverse=True)
    
    print("\n🏆 RANKING:")
    for i, (nome, r) in enumerate(ranking, 1):
        print(f"\n{i}. {nome}")
        print(f"   Discriminação: {r['discriminacao']:.4f}")
        print(f"   Relacionados: {r['media_relacionados']:.4f}")
        print(f"   Não relacionados: {r['media_nao_relacionados']:.4f}")
        print(f"   Tempo carga: {r['tempo_carga']:.2f}s")
    
    return ranking[0][0] if ranking else None

if __name__ == "__main__":
    arquivos = listar_resultados()
    
    if arquivos:
        escolha = input(f"\nDigite o número do resultado para ver detalhes (1-{len(arquivos)}): ")
        try:
            idx = int(escolha) - 1
            if 0 <= idx < len(arquivos):
                melhor = carregar_resultado(arquivos[idx])
                if melhor:
                    print(f"\n🎯 Melhor modelo neste teste: {melhor}")
        except ValueError:
            print("Opção inválida")