"""Database manager to encapsulate all global state and database operations."""

from typing import Dict, List, Any, Optional
import numpy as np
import faiss

from config import DatabaseConfig
from models import ValidationError, DatabaseError, CompanyNotFoundError
from database_core import (
    append_to_db, get_db_length, get_last_index, is_valid_index,
    get_item_by_index, copy_db
)


class DatabaseManager:
    """Encapsulates all database state and provides controlled access."""
    
    def __init__(self):
        """Initialize empty databases and indices."""
        # Core databases
        self._company_db: List[Dict[str, Any]] = []
        self._vector_db: List[np.ndarray] = []
        self._needs_vector_db: List[np.ndarray] = []
        
        # FAISS indices
        self._index: Optional[faiss.IndexFlatL2] = None
        self._needs_index: Optional[faiss.IndexFlatL2] = None
        self._index_dirty = False
        
        # Initialize FAISS indices
        self._initialize_indices()
    
    def _initialize_indices(self):
        """Initialize FAISS indices with correct dimensions."""
        try:
            from embedding import dimension
            self._index = faiss.IndexFlatL2(dimension)
            self._needs_index = faiss.IndexFlatL2(dimension)
        except Exception as e:
            from models import EmbeddingError
            raise EmbeddingError(f"Failed to initialize FAISS indices: {str(e)}")
    
    # === COMPANY DATABASE OPERATIONS ===
    
    def add_company(self, company_data: Dict[str, Any]) -> int:
        """Add company to database and return index."""
        if not isinstance(company_data, dict):
            raise ValidationError("Company data must be a dictionary")
        
        # Validate all required fields are present
        for field in DatabaseConfig.REQUIRED_FIELDS:
            if field not in company_data:
                raise ValidationError(f"Missing required field: {field}")
        
        try:
            append_to_db(company_data, self._company_db)
            return get_last_index(self._company_db)
        except Exception as e:
            raise DatabaseError(f"Failed to add company: {str(e)}")
    
    def get_company(self, index: int) -> Optional[Dict[str, Any]]:
        """Get company by index."""
        try:
            return get_item_by_index(index, self._company_db)
        except Exception as e:
            raise DatabaseError(f"Failed to get company: {str(e)}")
    
    def get_all_companies(self) -> List[Dict[str, Any]]:
        """Get all companies."""
        try:
            return copy_db(self._company_db)
        except Exception as e:
            raise DatabaseError(f"Failed to get all companies: {str(e)}")
    
    def get_company_count(self) -> int:
        """Get total number of companies."""
        try:
            return get_db_length(self._company_db)
        except Exception as e:
            raise DatabaseError(f"Failed to get company count: {str(e)}")
    
    def update_company(self, index: int, company_data: Dict[str, Any]) -> bool:
        """Update company at given index."""
        if not isinstance(index, int):
            raise ValidationError("Index must be an integer")
        
        if not isinstance(company_data, dict):
            raise ValidationError("Company data must be a dictionary")
        
        if not is_valid_index(index, self._company_db):
            raise CompanyNotFoundError(f"Company at index {index} not found")
        
        # Validate all required fields are present
        for field in DatabaseConfig.REQUIRED_FIELDS:
            if field not in company_data:
                raise ValidationError(f"Missing required field: {field}")
        
        try:
            # Update company data
            self._company_db[index] = company_data
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to update company: {str(e)}")
    
    def delete_company(self, index: int) -> bool:
        """Delete company at given index."""
        if not isinstance(index, int):
            raise ValidationError("Index must be an integer")
        
        if not is_valid_index(index, self._company_db):
            raise CompanyNotFoundError(f"Company at index {index} not found")
        
        try:
            # Remove from company database
            self._company_db.pop(index)
            
            # Remove from vector databases if they exist at this index
            if index < len(self._vector_db):
                self._vector_db.pop(index)
            if index < len(self._needs_vector_db):
                self._needs_vector_db.pop(index)
            
            # Mark indices for rebuild
            self.mark_indices_dirty()
            
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to delete company: {str(e)}")
    
    # === VECTOR DATABASE OPERATIONS ===
    
    def add_embedding(self, embedding: np.ndarray) -> int:
        \"\"\"Add embedding to main vector database.\"\"\"
        if not isinstance(embedding, np.ndarray):
            raise ValidationError("Embedding must be a numpy array")
        
        self._vector_db.append(embedding)
        
        # Add to FAISS index
        from embedding import convert_to_numpy_array
        embedding_array = convert_to_numpy_array(embedding)
        self._index.add(embedding_array)
        
        return len(self._vector_db) - 1
    
    def add_needs_embedding(self, embedding: np.ndarray) -> int:
        \"\"\"Add embedding to needs vector database.\"\"\"
        if not isinstance(embedding, np.ndarray):
            raise ValidationError("Embedding must be a numpy array")
        
        self._needs_vector_db.append(embedding)
        
        # Add to FAISS index
        from embedding import convert_to_numpy_array
        embedding_array = convert_to_numpy_array(embedding)
        self._needs_index.add(embedding_array)
        
        return len(self._needs_vector_db) - 1
    
    def update_embeddings(self, index: int, desc_embedding: np.ndarray, needs_embedding: np.ndarray):
        \"\"\"Update embeddings at given index.\"\"\"
        # Ensure vector databases are the right size
        while len(self._vector_db) <= index:
            self._vector_db.append(None)
        while len(self._needs_vector_db) <= index:
            self._needs_vector_db.append(None)
            
        self._vector_db[index] = desc_embedding
        self._needs_vector_db[index] = needs_embedding
        
        # Mark indices for rebuild
        self.mark_indices_dirty()
    
    def get_vector_count(self) -> int:
        \"\"\"Get number of vectors in main database.\"\"\"
        return len(self._vector_db)
    
    def get_needs_vector_count(self) -> int:
        \"\"\"Get number of vectors in needs database.\"\"\"
        return len(self._needs_vector_db)
    
    # === FAISS INDEX OPERATIONS ===
    
    def get_main_index(self) -> faiss.IndexFlatL2:
        \"\"\"Get main FAISS index (ensure it's current first).\"\"\"
        self.ensure_indices_current()
        return self._index
    
    def get_needs_index(self) -> faiss.IndexFlatL2:
        \"\"\"Get needs FAISS index (ensure it's current first).\"\"\"
        self.ensure_indices_current()
        return self._needs_index
    
    def mark_indices_dirty(self):
        \"\"\"Mark indices as needing rebuild.\"\"\"
        self._index_dirty = True
    
    def rebuild_indices(self) -> None:
        \"\"\"Rebuild both FAISS indices from current vector databases.\"\"\"
        try:
            from embedding import dimension, convert_to_numpy_array
            
            # Rebuild main index (description + challenges)
            self._index = faiss.IndexFlatL2(dimension)
            for vec in self._vector_db:
                if vec is not None:
                    embedding_array = convert_to_numpy_array(vec)
                    self._index.add(embedding_array)
            
            # Rebuild needs index
            self._needs_index = faiss.IndexFlatL2(dimension)
            for vec in self._needs_vector_db:
                if vec is not None:
                    embedding_array = convert_to_numpy_array(vec)
                    self._needs_index.add(embedding_array)
            
            self._index_dirty = False
        except Exception as e:
            from models import EmbeddingError
            raise EmbeddingError(f"Failed to rebuild FAISS indices: {str(e)}")
    
    def ensure_indices_current(self):
        \"\"\"Rebuild indices if they are marked as dirty.\"\"\"
        if self._index_dirty:
            self.rebuild_indices()
    
    # === UTILITY METHODS ===
    
    def clear_all_data(self):
        \"\"\"Clear all databases (useful for testing).\"\"\"
        self._company_db.clear()
        self._vector_db.clear()
        self._needs_vector_db.clear()
        self._initialize_indices()
        self._index_dirty = False
    
    def get_all_indices(self) -> List[int]:
        \"\"\"Get list of all valid company indices.\"\"\"
        return list(range(self.get_company_count()))