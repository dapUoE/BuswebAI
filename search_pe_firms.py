#!/usr/bin/env python3
"""
Search PE firms using vector similarity.

Usage: python search_pe_firms.py "your search query"

Returns top 30 most similar PE firms with similarity scores.
"""

import sys
import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PEFirmSearcher:
    def __init__(self, db_path='pe_firms.db'):
        self.db_path = db_path
        self.index_file = db_path.replace('.db', '_index.faiss')
        self.ids_file = db_path.replace('.db', '_ids.npy')
        self.conn = None
        self.cursor = None
        self.model = None
        self.faiss_index = None
        self.firm_ids = None
        
    def __enter__(self):
        self._connect_db()
        self._load_index()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def _connect_db(self):
        """Connect to SQLite database"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        logger.debug("Connected to database")
    
    def _load_index(self):
        """Load FAISS index and ID mappings"""
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"FAISS index not found: {self.index_file}")
        
        if not os.path.exists(self.ids_file):
            raise FileNotFoundError(f"ID mapping not found: {self.ids_file}")
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(self.index_file)
        
        # Load firm IDs
        self.firm_ids = np.load(self.ids_file)
        
        logger.info(f"Loaded index with {self.faiss_index.ntotal} embeddings")
    
    def _load_model(self):
        """Load sentence transformer model"""
        if self.model is None:
            logger.info("Loading embedding model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Model loaded successfully")
    
    def _create_embedding(self, text):
        """Create embedding for text"""
        self._load_model()
        return self.model.encode(text, convert_to_numpy=True)
    
    def search(self, query, top_k=30):
        """Search for PE firms similar to query"""
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Searching for: '{query}'")
        
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Search FAISS index
        search_k = min(top_k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(
            np.array([query_embedding], dtype=np.float32), 
            search_k
        )
        
        logger.debug(f"FAISS returned {len(indices[0])} indices")
        logger.debug(f"Distance range: {distances[0].min():.4f} - {distances[0].max():.4f}")
        
        # Get firm details from database
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            # Skip invalid indices
            if idx == -1 or idx >= len(self.firm_ids):
                logger.debug(f"Skipping invalid index: {idx}")
                continue
                
            firm_id = self.firm_ids[idx]
            firm_data = self._get_firm_by_id(firm_id)
            
            if firm_data:
                # Convert L2 distance to similarity score (0-1, higher is better)
                similarity = 1.0 / (1.0 + distance)
                results.append({
                    'id': firm_id,
                    'name': firm_data[0],
                    'description': firm_data[1],
                    'similarity': round(similarity, 4),
                    'distance': round(float(distance), 4)
                })
                logger.debug(f"Added result {i+1}: {firm_data[0][:30]}... (similarity: {similarity:.4f})")
            else:
                logger.debug(f"No firm data found for ID: {firm_id}")
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def _get_firm_by_id(self, firm_id):
        """Get firm data by ID"""
        try:
            self.cursor.execute('''
                SELECT name, description FROM pe_firms WHERE id = ?
            ''', (int(firm_id),))
            result = self.cursor.fetchone()
            if result:
                logger.debug(f"Found firm ID {firm_id}: {result[0]}")
            else:
                logger.debug(f"No firm found for ID {firm_id}")
            return result
        except Exception as e:
            logger.error(f"Database error for ID {firm_id}: {e}")
            return None
    
    def get_total_firms(self):
        """Get total number of firms in database"""
        self.cursor.execute("SELECT COUNT(*) FROM pe_firms")
        return self.cursor.fetchone()[0]

def main():
    if len(sys.argv) != 2:
        print("Usage: python search_pe_firms.py \"your search query\"")
        print("Example: python search_pe_firms.py \"technology investments\"")
        return 1
    
    query = sys.argv[1]
    
    # Enable debug logging if query starts with 'debug:'
    if query.startswith('debug:'):
        logging.getLogger().setLevel(logging.DEBUG)
        query = query[6:].strip()
    
    try:
        with PEFirmSearcher() as searcher:
            total_firms = searcher.get_total_firms()
            print(f"Searching {total_firms} PE firms for: '{query}'\n")
            
            results = searcher.search(query, top_k=30)
            
            if results:
                print(f"Top {len(results)} PE firms for '{query}':\n")
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['name']}")
            else:
                print("No results found.")
                print("Try using 'debug:your query' to see more details.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())