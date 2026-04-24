import os
import time
import json
import sys
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. CONFIGURAÇÃO DE ESTADO (CHECKPOINTING)
# ==========================================
#STATE_FILE = "state_acenario_a.json"
STATE_FILE = "state_acenario_b.json"

def carregar_estado():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            return state.get("last_processed_index", 0)
    return 0

def salvar_estado(index):
    with open(STATE_FILE, 'w') as f:
        json.dump({"last_processed_index": index}, f)

# ==========================================
# 2. DEFINIÇÃO DO LLM E PROMPT
# ==========================================
class NotasRelevancia(BaseModel):
    relevance_annotation_1: int = Field(description="Nota de 0 a 3 para o doc 1")
    relevance_annotation_2: int = Field(description="Nota de 0 a 3 para o doc 2")
    relevance_annotation_3: int = Field(description="Nota de 0 a 3 para o doc 3")
    relevance_annotation_4: int = Field(description="Nota de 0 a 3 para o doc 4")

llm = ChatGroq(
    temperature=0, 
    model_name=  "llama-3.3-70b-versatile", #"llama-3.1-8b-instant", 
    api_key=os.getenv("GROQ_API_KEY"),
    max_retries=3, 
    timeout=None,
)

llm_estruturado = llm.with_structured_output(NotasRelevancia)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um juiz avaliador especialista na área médica (saúde da mulher na menopausa).
Sua tarefa é avaliar se os documentos recuperados são relevantes para responder à pergunta do usuário.
Regras:

- Baseie-se apenas na pergunta.
- Pergunte: o documento responde diretamente ao que foi perguntado?
- Se não responder diretamente, não pode receber nota 3.
- Texto apenas relacionado ao tema, mas sem responder → nota 1 ou 0.
- Similaridade de assunto não implica relevância.
- Conteúdo genérico ou tangencial deve ser penalizado.
Use a seguinte escala rigorosa de 0 a 3:
3: Altamente relevante (Responde diretamente à pergunta)
2: Parcialmente relevante (Parcialmente útil, incompleto ou indireto)
1: Marginalmente relevante (Relacionado ao tema, mas não responde)
0: Irrelevante (Não responde à pergunta)"""),
    ("human", """
Pergunta do usuário: {question}

---
Documento 1: {documento_1}
---
Documento 2: {documento_2}
---
Documento 3: {documento_3}
---
Documento 4: {documento_4}
---

Avalie cada um dos 4 documentos acima e retorne as 4 notas.""")
])

chain = prompt | llm_estruturado

# ==========================================
# 3. LOOP PRINCIPAL DE AVALIAÇÃO
# ==========================================
def avaliar_dataset(arquivo_entrada, arquivo_saida):
    
    # === SISTEMA DE RETOMADA INTELIGENTE ===
    if os.path.exists(arquivo_saida):
        print(f"🔄 Arquivo de saída encontrado! Carregando histórico de: {arquivo_saida}")
        df = pd.read_csv(arquivo_saida, sep=';')
    else:
        print(f"📄 Iniciando nova avaliação. Carregando dados de: {arquivo_entrada}")
        df = pd.read_csv(arquivo_entrada, sep=';')
    
    colunas_notas = [
        'relevance_annotation_1', 'relevance_annotation_2', 
        'relevance_annotation_3', 'relevance_annotation_4'
    ]
    for col in colunas_notas:
        if col not in df.columns:
            df[col] = pd.NA

    total_linhas = len(df)
    start_index = carregar_estado()
    
    if start_index >= total_linhas:
        print("Todas as linhas já foram processadas! Apague o arquivo .json se quiser recomeçar.")
        return

    print(f"Retomando a partir da linha {start_index} de {total_linhas}...")

    for index in range(start_index, total_linhas):
        row = df.iloc[index]
        print(f"Avaliando pergunta {index + 1}/{total_linhas}...")
        
        sucesso = False
        
        while not sucesso:
            try:
                # Usa as chaves exatas do seu ficheiro original (documento_1, etc.)
                resultado = chain.invoke({
                    "question": row['question'],
                    "documento_1": row.get('documento_1', 'Vazio') or 'Vazio',
                    "documento_2": row.get('documento_2', 'Vazio') or 'Vazio',
                    "documento_3": row.get('documento_3', 'Vazio') or 'Vazio',
                    "documento_4": row.get('documento_4', 'Vazio') or 'Vazio'
                })
                
                df.at[index, 'relevance_annotation_1'] = resultado.relevance_annotation_1
                df.at[index, 'relevance_annotation_2'] = resultado.relevance_annotation_2
                df.at[index, 'relevance_annotation_3'] = resultado.relevance_annotation_3
                df.at[index, 'relevance_annotation_4'] = resultado.relevance_annotation_4
                
                # Força as colunas de notas a serem Inteiros (evita o 0.0)
                df[colunas_notas] = df[colunas_notas].astype('Int64')
                
                df.to_csv(arquivo_saida, sep=';', index=False)
                salvar_estado(index + 1)
                
                sucesso = True 
                time.sleep(1.5) 

            except Exception as e:
                erro_msg = str(e).lower()
                
                if "rate limit" in erro_msg or "429" in erro_msg or "exceeded" in erro_msg:
                    print(f"\n🛑 Limite da API Groq atingido na linha {index}!")
                    print("A execução foi interrompida conforme programado.")
                    print("Rode este script novamente mais tarde para continuar exatamente de onde parou.")
                    return 
                else:
                    print(f"❌ Erro inesperado na linha {index}: {e}")
                    print("Aguardando 10 segundos para tentar novamente...")
                    time.sleep(10)

    print(f"\n✅ Avaliação 100% concluída! Resultados salvos em: {arquivo_saida}")

# --- COMO EXECUTAR ---
if __name__ == "__main__":
    #arquivo_entrada = "result_scenario_a_baseline_lexica.csv" 
    #arquivo_saida = "relevance_anotation_scenario_a_baseline_lexica.csv"
    
    arquivo_entrada = "result_scenario_b_baseline_dense.csv" 
    arquivo_saida = "relevance_anotation_scenario_b_baseline_dense.csv"
    avaliar_dataset(arquivo_entrada, arquivo_saida)