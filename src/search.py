"""Search functionality for the company database system."""

from typing import Dict, List, Any, Optional
import numpy as np
import faiss

from config import DatabaseConfig
from models import ValidationError, DatabaseError, EmbeddingError
from embedding import (
    encode_text, convert_to_numpy_array, search_faiss_index, 
    get_first_distances, get_first_indices, round_score
)
from database_core import get_company, get_name, get_description, get_needs, get_challenges, get_website

# === GLOBAL STATE MANAGEMENT ===
# In-memory database
company_db = []  # List of dicts for metadata
vector_db = []   # List of embedding vectors for (description + challenges)
needs_vector_db = []  # List of embedding vectors for needs only

# FAISS indices
index = None
needs_index = None
_index_dirty = False  # Flag to track if indices need rebuilding

def initialize_indices():
    """Initialize FAISS indices"""
    global index, needs_index
    try:
        from embedding import dimension
        index = faiss.IndexFlatL2(dimension)
        needs_index = faiss.IndexFlatL2(dimension)
    except Exception as e:
        raise EmbeddingError(f"Failed to initialize indices: {str(e)}")

def mark_indices_dirty():
    """Mark indices as needing rebuild"""
    global _index_dirty
    _index_dirty = True

def rebuild_faiss_index() -> None:
    """Rebuild both FAISS indices from current vector databases"""
    global index, needs_index, _index_dirty
    try:
        from embedding import dimension, add_embedding_to_index
        
        # Rebuild main index (description + challenges)
        index = faiss.IndexFlatL2(dimension)
        for vec in vector_db:
            embedding_array = convert_to_numpy_array(vec)
            index.add(embedding_array)
        
        # Rebuild needs index
        needs_index = faiss.IndexFlatL2(dimension)
        for vec in needs_vector_db:
            embedding_array = convert_to_numpy_array(vec)
            needs_index.add(embedding_array)
        
        _index_dirty = False
    except Exception as e:
        raise EmbeddingError(f"Failed to rebuild FAISS indices: {str(e)}")

def ensure_indices_current():
    """Rebuild indices if they are marked as dirty"""
    if _index_dirty:
        rebuild_faiss_index()

# === SEARCH FUNCTIONS ===
def search_embeddings(query_text: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> tuple:
    """Search embeddings with query text"""
    if not isinstance(query_text, str):
        raise ValidationError("Query text must be a string")
    
    if not query_text.strip():
        raise ValidationError("Query text cannot be empty")
    
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValidationError("top_k must be a positive integer")
    
    try:
        ensure_indices_current()  # Rebuild if needed before search
        query_vec = encode_text(query_text)
        query_array = convert_to_numpy_array(query_vec)
        search_result = search_faiss_index(query_array, top_k, index)
        distances = get_first_distances(search_result)
        indices = get_first_indices(search_result)
        return distances, indices
    except Exception as e:
        raise EmbeddingError(f"Failed to search embeddings: {str(e)}")

def search_needs_embeddings(query_text: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> tuple:
    """Search needs embeddings with query text"""
    if not isinstance(query_text, str):
        raise ValidationError("Query text must be a string")
    
    if not query_text.strip():
        raise ValidationError("Query text cannot be empty")
    
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValidationError("top_k must be a positive integer")
    
    try:
        ensure_indices_current()  # Rebuild if needed before search
        query_vec = encode_text(query_text)
        query_array = convert_to_numpy_array(query_vec)
        search_result = search_faiss_index(query_array, top_k, needs_index)
        distances = get_first_distances(search_result)
        indices = get_first_indices(search_result)
        return distances, indices
    except Exception as e:
        raise EmbeddingError(f"Failed to search needs embeddings: {str(e)}")

def create_search_result(company: Dict[str, Any], score: float) -> Dict[str, Any]:
    """Create search result dictionary"""
    try:
        return {
            "name": get_name(company),
            "match_score": round_score(score),
            "description": get_description(company),
            "needs": get_needs(company),
            "challenges": get_challenges(company),
            "website": get_website(company)
        }
    except Exception as e:
        raise DatabaseError(f"Failed to create search result: {str(e)}")

def search_companies_by_text(query: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Search companies by text query"""
    try:
        distances, indices = search_embeddings(query, top_k)
        
        results = []
        for idx, score in zip(indices, distances):
            # Convert numpy int to Python int
            idx = int(idx)
            score = float(score)
            match = get_company(idx, company_db)
            if match:
                search_result = create_search_result(match, score)
                results.append(search_result)
        return results
    except Exception as e:
        raise DatabaseError(f"Failed to search companies: {str(e)}")

def search_companies_by_needs(query: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Search companies by their needs specifically"""
    try:
        distances, indices = search_needs_embeddings(query, top_k)
        
        results = []
        for idx, score in zip(indices, distances):
            # Convert numpy int to Python int
            idx = int(idx)
            score = float(score)
            match = get_company(idx, company_db)
            if match:
                search_result = create_search_result(match, score)
                results.append(search_result)
        return results
    except Exception as e:
        raise DatabaseError(f"Failed to search companies by needs: {str(e)}")

# Initialize indices when module is imported
initialize_indices()