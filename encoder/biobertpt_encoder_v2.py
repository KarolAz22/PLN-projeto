"""
Encoder BioBERTpt com Mean Pooling (melhor para similaridade)
"""

import torch
import numpy as np
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioBERTptEncoderV2:
    """Encoder com Mean Pooling para melhores embeddings"""
    
    def __init__(
        self,
        model_name: str = "pucpr/biobertpt-all",
        device: Optional[str] = None,
        max_length: int = 512,
        pooling: str = "mean"  # 'mean', 'cls', 'max'
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        
        # Define dispositivo
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"📱 Dispositivo: {self.device}")
        logger.info(f"🎯 Pooling strategy: {pooling}")
        
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
    
    def _mean_pooling(self, embeddings, attention_mask):
        """Mean Pooling - média dos tokens (ignorando padding)"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _cls_pooling(self, embeddings):
        """CLS Pooling - primeiro token (padrão BERT)"""
        return embeddings[:, 0, :]
    
    def _max_pooling(self, embeddings, attention_mask):
        """Max Pooling - máximo dos tokens"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        embeddings = embeddings.clone()
        embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(embeddings, 1)[0]
    
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
        outputs = self.model(**tokens)
        
        # Aplica pooling
        if self.pooling == "mean":
            embedding = self._mean_pooling(outputs.last_hidden_state, tokens['attention_mask'])
        elif self.pooling == "max":
            embedding = self._max_pooling(outputs.last_hidden_state, tokens['attention_mask'])
        else:  # cls
            embedding = self._cls_pooling(outputs.last_hidden_state)
        
        # Move para CPU e normaliza
        embedding = embedding.cpu().numpy()[0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    @torch.no_grad()
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
        outputs = self.model(**tokens)
        
        # Aplica pooling
        if self.pooling == "mean":
            embeddings = self._mean_pooling(outputs.last_hidden_state, tokens['attention_mask'])
        elif self.pooling == "max":
            embeddings = self._max_pooling(outputs.last_hidden_state, tokens['attention_mask'])
        else:  # cls
            embeddings = self._cls_pooling(outputs.last_hidden_state)
        
        embeddings = embeddings.cpu().numpy()
        
        # Normaliza cada embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    @staticmethod
    def similaridade_cosseno(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcula similaridade entre dois embeddings"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))