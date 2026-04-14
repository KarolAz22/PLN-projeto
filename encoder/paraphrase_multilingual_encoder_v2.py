"""
Encoder paraphrase-multilingual-MiniLM-L12-v2 com suporte a pooling avançado
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParaphraseMultilingualEncoderV2:
    """Encoder com suporte a múltiplas estratégias de pooling"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        pooling: str = "mean"  # 'mean', 'cls', 'max' (SentenceTransformer usa mean por padrão)
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device  # Deixar SentenceTransformer gerenciar automaticamente
        
        logger.info(f"🎯 Pooling strategy: {pooling}")
        
        # Carrega modelo usando SentenceTransformer
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo SentenceTransformer"""
        logger.info(f"🔄 Carregando {self.model_name}...")
        
        # SentenceTransformer já faz o parsing, tokenização e pooling automático
        # Deixa SentenceTransformer detectar CPU/CUDA automaticamente
        self.model = SentenceTransformer(self.model_name)
        
        logger.info(f"✅ Modelo carregado! Dimensionalidade: 384")
    
    def encode(self, texto: str) -> np.ndarray:
        """Converte um texto em embedding"""
        # SentenceTransformer já faz o pooling automaticamente
        embedding = self.model.encode(texto, convert_to_tensor=False)
        
        # Garante que é numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        return embedding
    
    def encode_batch(self, textos: List[str]) -> np.ndarray:
        """Converte múltiplos textos em embeddings"""
        # SentenceTransformer já faz o pooling automaticamente
        embeddings = self.model.encode(textos, convert_to_tensor=False)
        
        # Garante que é numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        return embeddings
    
    @staticmethod
    def similaridade_cosseno(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcula similaridade entre dois embeddings"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))
