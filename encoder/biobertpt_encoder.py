"""
Encoder para BioBERTpt CLS pooling - APENAS para USAR o modelo (não treina)
"""

import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioBERTptEncoder:
    """Encoder para gerar embeddings usando BioBERTpt"""
    
    def __init__(
        self,
        model_name: str = "pucpr/biobertpt-all",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.max_length = max_length
        
        # Define dispositivo
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
"cpu")                                                                                  
        logger.info(f"📱 Dispositivo: {self.device}")
        
        # Carrega modelo
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo e tokenizer"""
        logger.info(f"🔄 Carregando {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Modelo carregado! Vocabulário: {len(self.tokenizer)}")
    
    @torch.no_grad()
    def encode(self, texto: str) -> np.ndarray:
        """Converte um texto em embedding"""
        # Tokeniza
        tokens = self.tokenizer(
            texto,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        
        # Move para dispositivo
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Forward pass
        saida = self.model(**tokens)
        
        # Pega embedding do token [CLS]
        embedding = saida.last_hidden_state[:, 0, :].cpu().detach().numpy()[0]
        
        # Normaliza
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def encode_batch(self, textos: List[str]) -> np.ndarray:
        """Converte múltiplos textos em embeddings"""
        tokens = self.tokenizer(
            textos,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        saida = self.model(**tokens)
        
        embeddings = saida.last_hidden_state[:, 0, :].cpu().detach().numpy()
        
        # Normaliza cada embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    @staticmethod
    def similaridade_cosseno(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcula similaridade entre dois embeddings"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm
(emb2)))                                                                        

if __name__ == "__main__":
    print("=" * 50)
    print("TESTE RÁPIDO DO ENCODER")
    print("=" * 50)
    
    encoder = BioBERTptEncoder()
    
    # Teste 1: texto único
    emb = encoder.encode("menopausa")
    print(f"\n✅ Texto único: menopausa")
    print(f"   Shape: {emb.shape}")
    print(f"   Norma: {np.linalg.norm(emb):.4f}")
    
    # Teste 2: batch
    textos = ["menopausa", "fogacho", "estrogênio"]
    embs = encoder.encode_batch(textos)
    print(f"\n✅ Batch com {len(textos)} textos")
    print(f"   Shape: {embs.shape}")
    
    # Teste 3: similaridade
    sim = encoder.similaridade_cosseno(embs[0], embs[1])
    print(f"\n✅ Similaridade 'menopausa' ↔ 'fogacho': {sim:.4f}")