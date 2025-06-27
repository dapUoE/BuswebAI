#!/usr/bin/env python3
"""
Search PE firms using tag-based vector similarity.

Usage: python search_pe_firms_tags.py "your search query"

Returns top 30 most similar PE firms based on tag similarity.
The search query is converted to tags and then matched against firm tag embeddings.
"""

import sys
import os
import sqlite3
import numpy as np
import faiss
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tag_embedding_service import TagEmbeddingService
from models import ValidationError, EmbeddingError


class PEFirmTagSearcher:
    def __init__(self, db_path='pe_firms_tags.db'):
        self.db_path = db_path
        self.index_file = db_path.replace('.db', '_index.faiss')
        self.ids_file = db_path.replace('.db', '_ids.npy')
        self.conn = None
        self.cursor = None
        self.tag_service = None
        self.faiss_index = None
        self.firm_ids = None
        
    def __enter__(self):
        self._connect_db()
        self._load_index()
        self._init_tag_service()
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
        
        logger.info(f"Loaded tag-based index with {self.faiss_index.ntotal} embeddings")
    
    def _init_tag_service(self):
        """Initialize tag embedding service"""
        try:
            self.tag_service = TagEmbeddingService()
            logger.debug("Tag embedding service initialized")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize tag service: {str(e)}")
    
    def search_by_description(self, query_description: str, top_k=30):
        """
        Search for PE firms similar to a query description.
        The query is converted to tags first, then embedded.
        """
        if not query_description.strip():
            raise ValueError("Query description cannot be empty")
        
        logger.info(f"Searching for: '{query_description}'")
        
        try:
            # Step 1: Generate tags from query description
            logger.debug("Generating tags from query description...")
            query_tags, query_embedding = self.tag_service.generate_tags_and_embedding(query_description)
            
            # Log the generated tags
            tag_string = self.tag_service.tag_generator.tags_to_string(query_tags)
            logger.info(f"Query tags: {tag_string}")
            
            # Step 2: Search using the tag embedding
            return self._search_with_embedding(query_embedding, query_tags, top_k)
            
        except Exception as e:
            raise EmbeddingError(f"Search failed: {str(e)}")
    
    def search_by_tags(self, tags: dict, top_k=30):
        """
        Search for PE firms similar to provided tags.
        """
        logger.info(f"Searching by tags: {self.tag_service.tag_generator.tags_to_string(tags)}")
        
        try:
            # Create embedding from tags
            query_embedding = self.tag_service.create_embedding_from_tags(tags)
            
            return self._search_with_embedding(query_embedding, tags, top_k)
            
        except Exception as e:
            raise EmbeddingError(f"Tag search failed: {str(e)}")
    
    def search_by_tag_string(self, tag_string: str, top_k=30):
        """
        Search for PE firms using a tag string directly.
        Format: "industry:fintech technology:machine-learning business_model:saas"
        """
        if not tag_string.strip():
            raise ValueError("Tag string cannot be empty")
        
        logger.info(f"Searching by tag string: '{tag_string}'")
        
        try:
            # Create embedding from tag string
            query_embedding = self.tag_service.create_embedding_from_tag_string(tag_string)
            
            return self._search_with_embedding(query_embedding, {"custom": [tag_string]}, top_k)
            
        except Exception as e:
            raise EmbeddingError(f"Tag string search failed: {str(e)}")
    
    def _search_with_embedding(self, query_embedding: np.ndarray, query_info: dict, top_k: int):
        """Internal method to search with a pre-computed embedding"""
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
                
                # Parse the stored tags
                stored_tags = json.loads(firm_data[2]) if firm_data[2] else {}
                
                results.append({
                    'id': firm_id,
                    'name': firm_data[0],
                    'description': firm_data[1],
                    'tags': stored_tags,
                    'tag_string': firm_data[3],
                    'similarity': round(similarity, 4),
                    'distance': round(float(distance), 4)
                })
                logger.debug(f"Added result {i+1}: {firm_data[0][:30]}... (similarity: {similarity:.4f})")
            else:
                logger.debug(f"No firm data found for ID: {firm_id}")
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def _get_firm_by_id(self, firm_id):
        """Get firm data by ID including tags"""
        try:
            self.cursor.execute('''
                SELECT name, description, tags, tag_string FROM pe_firms WHERE id = ?
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
    
    def get_firm_tags_by_id(self, firm_id: int):
        """Get detailed tags for a specific firm"""
        self.cursor.execute('SELECT name, tags, tag_string FROM pe_firms WHERE id = ?', (firm_id,))
        result = self.cursor.fetchone()
        if result:
            return {
                'name': result[0],
                'tags': json.loads(result[1]),
                'tag_string': result[2]
            }
        return None
    
    def get_tag_statistics(self):
        """Get statistics about all tags in the database"""
        self.cursor.execute('SELECT tags FROM pe_firms')
        rows = self.cursor.fetchall()
        
        category_counts = {
            "industry": {},
            "technology": {},
            "business_model": {},
            "stage": {},
            "market": {},
            "solution_type": {}
        }
        
        for row in rows:
            tags = json.loads(row[0])
            for category, category_tags in tags.items():
                if category in category_counts:
                    for tag in category_tags:
                        category_counts[category][tag] = category_counts[category].get(tag, 0) + 1
        
        return category_counts


def main():
    if len(sys.argv) != 2:
        print("Usage: python search_pe_firms_tags.py \"your search query\"")
        print("Example: python search_pe_firms_tags.py \"AI healthcare technology\"")
        print("\nSearch modes:")
        print("  - Description: 'AI healthcare startup using machine learning'")
        print("  - Tag string: 'industry:healthcare technology:ai business_model:b2b'")
        return 1
    
    query = sys.argv[1]
    
    # Enable debug logging if query starts with 'debug:'
    if query.startswith('debug:'):
        logging.getLogger().setLevel(logging.DEBUG)
        query = query[6:].strip()
    
    try:
        with PEFirmTagSearcher() as searcher:
            total_firms = searcher.get_total_firms()
            print(f"🔍 Searching {total_firms} PE firms for: '{query}'\n")
            
            # Determine search mode based on query format
            if ':' in query and any(keyword in query for keyword in ['industry:', 'technology:', 'business_model:']):
                # Tag string search
                print("🏷️  Using tag string search mode")
                results = searcher.search_by_tag_string(query, top_k=30)
            else:
                # Description search (generate tags first)
                print("📝 Using description search mode (generating tags)")
                results = searcher.search_by_description(query, top_k=30)
            
            if results:
                print(f"\n🎯 Top {len(results)} PE firms for '{query}':\n")
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['name']} (similarity: {result['similarity']})")
                    print(f"   Tags: {result['tag_string']}")
                    if i <= 3:  # Show description for top 3
                        desc = result['description'][:100] + "..." if len(result['description']) > 100 else result['description']
                        print(f"   Description: {desc}")
                    print()
            else:
                print("❌ No results found.")
                print("💡 Try using 'debug:your query' to see more details.")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())