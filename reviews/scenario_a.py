import os
import json
import pandas as pd
import numpy as np
import time
import re
from rank_bm25 import BM25Okapi

# ==========================================
# 1. Configurações Iniciais e Parâmetros
# ==========================================
ARQUIVO_ENTRADA = "complete_balanced_dataset.csv"  
ARQUIVO_CHUNKS = "index/files/doc_chunks.jsonl" # <-- Ajuste este caminho se necessário
ARQUIVO_SAIDA_CENARIO_A = "result_scenario_a_baseline_lexica.csv"

print(f"Carregando dados de QA do arquivo '{ARQUIVO_ENTRADA}'...")
df_qa = pd.read_csv(ARQUIVO_ENTRADA)

# ==========================================
# 2. Carregamento e Indexação do Corpus (BM25)
# ==========================================
print(f"\n1️⃣ Carregando documentos do arquivo '{ARQUIVO_CHUNKS}'...")
corpus_textos = []

# Lê os trechos de texto diretamente do JSONL
with open(ARQUIVO_CHUNKS, "r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)
        texto = chunk.get("chunk_text", "")
        if texto:
            corpus_textos.append(texto)

print(f"Total de documentos carregados: {len(corpus_textos)}")

# Função simples de pré-processamento (Tokenização)
# Converte para minúsculas, remove pontuação e separa as palavras
def tokenizar(texto):
    texto = str(texto).lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto.split()

print("Tokenizando o corpus e inicializando o modelo BM25...")
start_time = time.time()
corpus_tokenizado = [tokenizar(doc) for doc in corpus_textos]
bm25 = BM25Okapi(corpus_tokenizado)
print(f"BM25 inicializado em {time.time() - start_time:.2f} segundos.")

# ==========================================
# 3. Busca Lexical (Extraindo Top 4)
# ==========================================
print(f"\n2️⃣ Iniciando a recuperação para {len(df_qa)} amostras (Cenário A)...")

docs_1, scores_1 = [], []
docs_2, scores_2 = [], []
docs_3, scores_3 = [], []
docs_4, scores_4 = [], []

tempo_busca = time.time()

for index, row in df_qa.iterrows():
    # Mantemos a busca usando a 'answer' para ficar metodologicamente idêntico ao Cenário B
    resposta_alvo = str(row['answer'])
    
    # Tokeniza a query
    query_tokenizada = tokenizar(resposta_alvo)
    
    # Obtém as pontuações de BM25 para TODOS os documentos do corpus
    pontuacoes_todas = bm25.get_scores(query_tokenizada)
    
    # Ordena os índices do maior score para o menor e pega os 4 primeiros
    # argsort retorna os índices em ordem crescente, então usamos [::-1] para inverter
    top_4_indices = np.argsort(pontuacoes_todas)[::-1][:4]
    
    # Inicializa vazios
    textos = ["", "", "", ""]
    pontuacoes = [0.0, 0.0, 0.0, 0.0]
    
    # Preenche com os top 4
    for i, idx_doc in enumerate(top_4_indices):
        textos[i] = corpus_textos[idx_doc]
        pontuacoes[i] = pontuacoes_todas[idx_doc]
        
    # Adiciona às listas finais
    docs_1.append(textos[0])
    scores_1.append(pontuacoes[0])
    
    docs_2.append(textos[1])
    scores_2.append(pontuacoes[1])
    
    docs_3.append(textos[2])
    scores_3.append(pontuacoes[2])
    
    docs_4.append(textos[3])
    scores_4.append(pontuacoes[3])
        
    if (index + 1) % 100 == 0:
        print(f"Processado: {index + 1}/{len(df_qa)}")

# ==========================================
# 4. Consolidação de Dados e Salvamento
# ==========================================
print(f"\nBusca com BM25 concluída em {time.time() - tempo_busca:.2f} segundos.")

# Insere os novos dados no dataframe original
df_qa['documento_1'] = docs_1
df_qa['score_mapeamento_1'] = scores_1

df_qa['documento_2'] = docs_2
df_qa['score_mapeamento_2'] = scores_2

df_qa['documento_3'] = docs_3
df_qa['score_mapeamento_3'] = scores_3

df_qa['documento_4'] = docs_4
df_qa['score_mapeamento_4'] = scores_4

print(f"\n3️⃣ Aplicando correções de formatação para o Excel Brasileiro...")

colunas_scores = [
    'score_mapeamento_1', 
    'score_mapeamento_2', 
    'score_mapeamento_3', 
    'score_mapeamento_4'
]

# Converte os números em strings e troca o ponto americano pela vírgula brasileira
# Nota: Os scores do BM25 costumam ser números maiores que 1 (ex: 14,582), 
# não são limitados entre 0 e 1 como a distância do cosseno.
for col in colunas_scores:
    df_qa[col] = df_qa[col].astype(str).str.replace('.', ',')

# Lista de colunas a serem exportadas
colunas_finais = [
    'question', 
    'answer', 
    'medical_specialty', 
    'question_type', 
    'is_augmented', 
    'risk', 
    'documento_1', 'score_mapeamento_1',
    'documento_2', 'score_mapeamento_2',
    'documento_3', 'score_mapeamento_3',
    'documento_4', 'score_mapeamento_4'
]

# Salva usando separador de ponto-e-vírgula (sep=';') para alinhar perfeitamente no Excel
df_qa[colunas_finais].to_csv(ARQUIVO_SAIDA_CENARIO_A, index=False, sep=';', encoding='utf-8')

print(f"Total de perguntas processadas e salvas: {len(df_qa)}")
print(f"✅ Arquivo '{ARQUIVO_SAIDA_CENARIO_A}' gerado com sucesso!")
print("Este arquivo representa o desempenho real do Cenário A (Baseline Lexical).")