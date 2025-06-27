"""Service for all embedding operations and AI model management."""

from typing import Optional
import numpy as np
from openai import OpenAI
import os

from config import DatabaseConfig
from models import ValidationError, EmbeddingError
from embedding import (
    encode_text, combine_text_blob, round_score,
    convert_to_numpy_array, search_faiss_index,
    get_first_distances, get_first_indices
)


class EmbeddingService:
    """Service for managing AI embeddings and model operations."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding service with optional custom model."""
        self._model_name = model_name or DatabaseConfig.OPENAI_EMBEDDING_MODEL
        self._client: Optional[OpenAI] = None
        self._dimension: Optional[int] = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                self._client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self._dimension = DatabaseConfig.EMBEDDING_DIMENSION
            except Exception as e:
                raise EmbeddingError(f"Failed to initialize OpenAI client: {str(e)}")
        return self._client
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (lazy-loaded)."""
        if self._dimension is None:
            _ = self.client  # Trigger client loading
        return self._dimension
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        """Create embedding from text."""
        try:
            return encode_text(text)
        except Exception as e:
            raise EmbeddingError(f"Failed to create embedding: {str(e)}")
    
    def create_description_embedding(self, description: str, challenges: str) -> np.ndarray:
        """Create embedding for company description + challenges."""
        try:
            combined_text = combine_text_blob(description, challenges)
            return self.create_text_embedding(combined_text)
        except Exception as e:
            raise EmbeddingError(f"Failed to create description embedding: {str(e)}")
    
    def create_needs_embedding(self, needs: str) -> np.ndarray:
        """Create embedding for company needs."""
        try:
            return self.create_text_embedding(needs)
        except Exception as e:
            raise EmbeddingError(f"Failed to create needs embedding: {str(e)}")
    
    def search_index(self, query_text: str, faiss_index, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> tuple:
        """Search FAISS index with query text."""
        if not isinstance(query_text, str):
            raise ValidationError("Query text must be a string")
        
        if not query_text.strip():
            raise ValidationError("Query text cannot be empty")
        
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValidationError("top_k must be a positive integer")
        
        try:
            # Create query embedding
            query_embedding = self.create_text_embedding(query_text)
            query_array = convert_to_numpy_array(query_embedding)
            
            # Search index
            search_result = search_faiss_index(query_array, top_k, faiss_index)
            distances = get_first_distances(search_result)
            indices = get_first_indices(search_result)
            
            return distances, indices
        except Exception as e:
            raise EmbeddingError(f"Failed to search index: {str(e)}")
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        try:
            emb1 = self.create_text_embedding(text1)
            emb2 = self.create_text_embedding(text2)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            raise EmbeddingError(f"Failed to calculate similarity: {str(e)}")
    
    def round_score(self, score: float, decimals: int = DatabaseConfig.DEFAULT_SCORE_DECIMALS) -> float:
        """Round similarity score to specified decimal places."""
        return round_score(score, decimals)