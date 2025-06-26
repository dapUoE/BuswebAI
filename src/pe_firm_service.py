"""
PE Firm service for managing private equity firms with vector search capabilities.
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from .models import DatabaseError, ValidationError, EmbeddingError
from .embedding_service import EmbeddingService
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class PEFirm:
    """PE Firm data structure with type hints and validation"""
    name: str
    description: str
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PEFirm to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PEFirm':
        """Create PEFirm from dictionary"""
        return cls(**data)

class PEFirmService:
    """Service for managing PE firms with vector search capabilities"""
    
    def __init__(self, db_manager: DatabaseManager = None, embedding_service: EmbeddingService = None):
        self.db_manager = db_manager or DatabaseManager()
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialize PE firm database table
        self._init_pe_firm_table()
    
    def _init_pe_firm_table(self):
        """Initialize the PE firm table in the database"""
        try:
            self.db_manager.cursor.execute('''
                CREATE TABLE IF NOT EXISTS pe_firms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL
                )
            ''')
            self.db_manager.conn.commit()
            logger.info("PE firm table initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PE firm table: {e}")
            raise DatabaseError(f"Failed to initialize PE firm table: {e}")
    
    def add_pe_firm(self, pe_firm: PEFirm) -> int:
        """Add a PE firm to the database with vector embedding"""
        try:
            # Validate input
            if not pe_firm.name or not pe_firm.description:
                raise ValidationError("PE firm name and description are required")
            
            # Create embedding for the description
            embedding = self.embedding_service.create_text_embedding(pe_firm.description)
            
            # Insert into database
            self.db_manager.cursor.execute('''
                INSERT INTO pe_firms (name, description) VALUES (?, ?)
            ''', (pe_firm.name, pe_firm.description))
            
            pe_firm_id = self.db_manager.cursor.lastrowid
            self.db_manager.conn.commit()
            
            # Add embedding to vector index
            self._add_pe_firm_embedding(pe_firm_id, embedding)
            
            logger.info(f"Added PE firm: {pe_firm.name} (ID: {pe_firm_id})")
            return pe_firm_id
            
        except Exception as e:
            logger.error(f"Failed to add PE firm {pe_firm.name}: {e}")
            raise DatabaseError(f"Failed to add PE firm: {e}")
    
    def _add_pe_firm_embedding(self, pe_firm_id: int, embedding: np.ndarray):
        """Add PE firm embedding to vector index"""
        try:
            # Ensure we have a PE firm vector index
            if not hasattr(self.db_manager, 'pe_firm_embeddings'):
                self.db_manager.pe_firm_embeddings = []
                self.db_manager.pe_firm_indices = []
                
                # Create FAISS index for PE firms
                import faiss
                dimension = len(embedding)
                self.db_manager.pe_firm_faiss_index = faiss.IndexFlatL2(dimension)
            
            # Add to our tracking lists
            self.db_manager.pe_firm_embeddings.append(embedding)
            self.db_manager.pe_firm_indices.append(pe_firm_id)
            
            # Add to FAISS index
            self.db_manager.pe_firm_faiss_index.add(np.array([embedding], dtype=np.float32))
            
            logger.info(f"Added PE firm embedding for ID: {pe_firm_id}")
            
        except Exception as e:
            logger.error(f"Failed to add PE firm embedding: {e}")
            raise EmbeddingError(f"Failed to add PE firm embedding: {e}")
    
    def search_pe_firms(self, query: str, top_k: int = 3) -> List[Tuple[PEFirm, float]]:
        """Search PE firms using vector similarity"""
        try:
            # Create query embedding
            query_embedding = self.embedding_service.create_text_embedding(query)
            
            # Check if we have any PE firms
            if not hasattr(self.db_manager, 'pe_firm_faiss_index') or self.db_manager.pe_firm_faiss_index.ntotal == 0:
                logger.warning("No PE firm embeddings available for search")
                return []
            
            # Search using FAISS
            distances, indices = self.db_manager.pe_firm_faiss_index.search(
                np.array([query_embedding], dtype=np.float32), top_k
            )
            
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.db_manager.pe_firm_indices):
                    pe_firm_id = self.db_manager.pe_firm_indices[idx]
                    pe_firm = self.get_pe_firm_by_id(pe_firm_id)
                    if pe_firm:
                        # Convert distance to similarity score (lower distance = higher similarity)
                        similarity = 1.0 / (1.0 + distance)
                        results.append((pe_firm, round(similarity, 3)))
            
            logger.info(f"Found {len(results)} PE firms for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search PE firms: {e}")
            raise DatabaseError(f"Failed to search PE firms: {e}")
    
    def get_pe_firm_by_id(self, pe_firm_id: int) -> Optional[PEFirm]:
        """Get a PE firm by ID"""
        try:
            self.db_manager.cursor.execute('''
                SELECT id, name, description FROM pe_firms WHERE id = ?
            ''', (pe_firm_id,))
            
            row = self.db_manager.cursor.fetchone()
            if row:
                return PEFirm(id=row[0], name=row[1], description=row[2])
            return None
            
        except Exception as e:
            logger.error(f"Failed to get PE firm by ID {pe_firm_id}: {e}")
            raise DatabaseError(f"Failed to get PE firm: {e}")
    
    def get_all_pe_firms(self) -> List[PEFirm]:
        """Get all PE firms"""
        try:
            self.db_manager.cursor.execute('''
                SELECT id, name, description FROM pe_firms ORDER BY name
            ''')
            
            rows = self.db_manager.cursor.fetchall()
            return [PEFirm(id=row[0], name=row[1], description=row[2]) for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get all PE firms: {e}")
            raise DatabaseError(f"Failed to get all PE firms: {e}")
    
    def add_pe_firms_from_list(self, pe_firms_data: List[Dict[str, str]]) -> List[int]:
        """Add multiple PE firms from a list of dictionaries"""
        added_ids = []
        
        for firm_data in pe_firms_data:
            try:
                pe_firm = PEFirm.from_dict(firm_data)
                firm_id = self.add_pe_firm(pe_firm)
                added_ids.append(firm_id)
            except Exception as e:
                logger.error(f"Failed to add PE firm {firm_data.get('name', 'Unknown')}: {e}")
                continue
        
        logger.info(f"Successfully added {len(added_ids)} PE firms out of {len(pe_firms_data)} provided")
        return added_ids
    
    def rebuild_pe_firm_index(self):
        """Rebuild the PE firm vector index"""
        try:
            # Get all PE firms
            pe_firms = self.get_all_pe_firms()
            
            if not pe_firms:
                logger.warning("No PE firms found to rebuild index")
                return
            
            # Reset embeddings
            self.db_manager.pe_firm_embeddings = []
            self.db_manager.pe_firm_indices = []
            
            # Create new FAISS index
            import faiss
            first_embedding = self.embedding_service.create_text_embedding(pe_firms[0].description)
            dimension = len(first_embedding)
            self.db_manager.pe_firm_faiss_index = faiss.IndexFlatL2(dimension)
            
            # Add all embeddings
            for pe_firm in pe_firms:
                embedding = self.embedding_service.create_text_embedding(pe_firm.description)
                self.db_manager.pe_firm_embeddings.append(embedding)
                self.db_manager.pe_firm_indices.append(pe_firm.id)
            
            # Add all embeddings to FAISS index at once
            embeddings_array = np.array(self.db_manager.pe_firm_embeddings, dtype=np.float32)
            self.db_manager.pe_firm_faiss_index.add(embeddings_array)
            
            logger.info(f"Rebuilt PE firm index with {len(pe_firms)} firms")
            
        except Exception as e:
            logger.error(f"Failed to rebuild PE firm index: {e}")
            raise DatabaseError(f"Failed to rebuild PE firm index: {e}")