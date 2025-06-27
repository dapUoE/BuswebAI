"""
Embedding operations for the company database system.
"""

import numpy as np
import faiss
from typing import List
from openai import OpenAI
import os

from config import DatabaseConfig
from models import ValidationError, EmbeddingError

# === MODEL SETUP ===
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    dimension = DatabaseConfig.EMBEDDING_DIMENSION
except Exception as e:
    raise EmbeddingError(f"Failed to initialize OpenAI client: {str(e)}")

# === EMBEDDING FUNCTIONS ===
def encode_text(text: str) -> np.ndarray:
    """Encode text to embedding with error handling"""
    if not isinstance(text, str):
        raise ValidationError("Text must be a string")
    
    if not text.strip():
        raise ValidationError("Text cannot be empty")
    
    try:
        response = client.embeddings.create(
            model=DatabaseConfig.OPENAI_EMBEDDING_MODEL,
            input=text
        )
        return np.array(response.data[0].embedding)
    except Exception as e:
        raise EmbeddingError(f"Failed to encode text: {str(e)}")

def convert_to_numpy_array(embedding: np.ndarray) -> np.ndarray:
    """Convert embedding to numpy array format for FAISS"""
    if not isinstance(embedding, np.ndarray):
        raise ValidationError("Embedding must be a numpy array")
    return np.array([embedding])

def create_embedding(text: str) -> np.ndarray:
    """Create embedding from text"""
    try:
        return encode_text(text)
    except Exception as e:
        raise EmbeddingError(f"Failed to create embedding: {str(e)}")

def add_embedding_to_index(embedding: np.ndarray, vector_list: List[np.ndarray], 
                          faiss_index: faiss.IndexFlatL2, db_type: str) -> int:
    """Generic function to add embedding to any vector database and FAISS index"""
    try:
        if not isinstance(embedding, np.ndarray):
            raise ValidationError("Embedding must be a numpy array")
        
        if embedding.shape != (dimension,):
            raise ValidationError(f"Embedding must have shape ({dimension},)")
        
        # Add to vector list
        vector_list.append(embedding)
        
        # Add to FAISS index
        embedding_array = convert_to_numpy_array(embedding)
        faiss_index.add(embedding_array)
        
        return len(vector_list) - 1
    except Exception as e:
        raise EmbeddingError(f"Failed to add {db_type} embedding: {str(e)}")

def search_faiss_index(query_array: np.ndarray, k: int, faiss_index: faiss.IndexFlatL2) -> tuple:
    """Search FAISS index"""
    if not isinstance(query_array, np.ndarray):
        raise ValidationError("Query array must be a numpy array")
    
    if not isinstance(k, int) or k <= 0:
        raise ValidationError("k must be a positive integer")
    
    try:
        return faiss_index.search(query_array, k)
    except Exception as e:
        raise EmbeddingError(f"Failed to search FAISS index: {str(e)}")

def get_first_distances(search_result: tuple) -> np.ndarray:
    """Get first distances from search result"""
    if not isinstance(search_result, tuple) or len(search_result) != 2:
        raise ValidationError("Search result must be a tuple of length 2")
    return search_result[0][0]

def get_first_indices(search_result: tuple) -> np.ndarray:
    """Get first indices from search result"""
    if not isinstance(search_result, tuple) or len(search_result) != 2:
        raise ValidationError("Search result must be a tuple of length 2")
    return search_result[1][0]

def combine_text_blob(description: str, challenges: str) -> str:
    """Combine description and challenges into search blob (needs stored separately)"""
    if not all(isinstance(x, str) for x in [description, challenges]):
        raise ValidationError("All text fields must be strings")
    return f"{description}. Challenges: {challenges}"

def round_score(score: float, decimals: int = DatabaseConfig.DEFAULT_SCORE_DECIMALS) -> float:
    """Round score to specified decimal places"""
    if not isinstance(score, (int, float)):
        raise ValidationError("Score must be a number")
    
    if not isinstance(decimals, int) or decimals < 0:
        raise ValidationError("Decimals must be a non-negative integer")
    
    return round(score, decimals)