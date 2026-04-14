"""
Encoder para paraphrase-multilingual-MiniLM-L12-v2 usando SentenceTransformer
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParaphraseMultilingualEncoder:
    """Encoder para gerar embeddings usando paraphrase-multilingual-MiniLM-L12-v2"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device  # Deixar SentenceTransformer gerenciar automaticamente
        
        logger.info(f"🔄 Carregando {self.model_name}...")
        
        # Carrega modelo usando SentenceTransformer
        # SentenceTransformer detecta automaticamente CUDA/CPU
        self.model = SentenceTransformer(model_name)
        
        logger.info(f"✅ Modelo carregado!")
    
    def encode(self, texto: str) -> np.ndarray:
        """Converte um texto em embedding"""
        # Usa SentenceTransformer para gerar embedding normalizado
        embedding = self.model.encode(texto, convert_to_tensor=False)
        
        # Garante que é numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        return embedding
    
    def encode_batch(self, textos: List[str]) -> np.ndarray:
        """Converte múltiplos textos em embeddings"""
        # Usa SentenceTransformer para gerar embeddings normalizados
        embeddings = self.model.encode(textos, convert_to_tensor=False)
        
        # Garante que é numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        return embeddings
    
    @staticmethod
    def similaridade_cosseno(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcula similaridade entre dois embeddings"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


if __name__ == "__main__":
    print("=" * 50)
    print("TESTE RÁPIDO DO ENCODER")
    print("=" * 50)
    
    encoder = ParaphraseMultilingualEncoder()
    
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
    
    print("\n🎯 Encoder funcionando perfeitamente!")
