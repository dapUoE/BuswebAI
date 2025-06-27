#!/usr/bin/env python3
"""
Load PE firms from CSV file into database with tag-based vector embeddings.

Usage: python load_pe_firms_tags.py pe_firms.csv

CSV format: company name, description

This version:
1. Generates structured tags from descriptions using ChatGPT
2. Creates embeddings from those tags instead of raw text
3. Stores both tags and embeddings for better similarity matching
"""

import sys
import os
import csv
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


class PEFirmTagLoader:
    def __init__(self, db_path='pe_firms_tags.db'):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.tag_service = None
        self.embeddings = []
        self.firm_ids = []
        self.faiss_index = None
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._init_database()
        self._init_tag_service()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def _init_database(self):
        """Initialize database tables with tag support"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pe_firms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,
                tags JSON NOT NULL,
                tag_string TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
        logger.info("Database initialized with tag support")
    
    def _init_tag_service(self):
        """Initialize the tag embedding service"""
        try:
            self.tag_service = TagEmbeddingService()
            logger.info("Tag embedding service initialized")
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize tag service: {str(e)}")
    
    def load_from_csv(self, csv_file):
        """Load PE firms from CSV file with tag generation"""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        logger.info(f"Loading PE firms from {csv_file} with tag generation")
        
        # Clear existing data
        self.cursor.execute("DELETE FROM pe_firms")
        self.conn.commit()
        
        added_count = 0
        skipped_count = 0
        tag_errors = 0
        
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Skip header if it exists
            first_row = next(csv_reader, None)
            if first_row and (first_row[0].lower() in ['company name', 'name', 'company'] or 
                             first_row[1].lower() in ['description', 'desc']):
                logger.info("Skipping header row")
            else:
                # Process first row if it's not a header
                if first_row and len(first_row) >= 2:
                    result = self._process_firm(first_row[0].strip(), first_row[1].strip(), 1)
                    if result == "added":
                        added_count += 1
                    elif result == "skipped":
                        skipped_count += 1
                    elif result == "tag_error":
                        tag_errors += 1
            
            # Process remaining rows
            for row_num, row in enumerate(csv_reader, start=2):
                if len(row) < 2:
                    logger.warning(f"Row {row_num}: Not enough columns, skipping")
                    skipped_count += 1
                    continue
                
                name = row[0].strip()
                description = row[1].strip()
                
                if not name or not description:
                    logger.warning(f"Row {row_num}: Empty name or description, skipping")
                    skipped_count += 1
                    continue
                
                result = self._process_firm(name, description, row_num)
                if result == "added":
                    added_count += 1
                    if added_count % 5 == 0:
                        logger.info(f"Processed {added_count} firms...")
                elif result == "skipped":
                    skipped_count += 1
                elif result == "tag_error":
                    tag_errors += 1
        
        logger.info(f"Completed: {added_count} firms added, {skipped_count} skipped, {tag_errors} tag generation errors")
        
        # Build FAISS index
        if self.embeddings:
            self._build_faiss_index()
        else:
            logger.warning("No embeddings to build index from")
        
        return added_count
    
    def _process_firm(self, name: str, description: str, row_num: int) -> str:
        """
        Process a single firm: generate tags, create embedding, store in database.
        
        Returns:
            "added" if successful
            "skipped" if failed due to validation
            "tag_error" if tag generation failed
        """
        try:
            # Generate tags and embedding
            logger.debug(f"Row {row_num}: Generating tags for {name}")
            tags, embedding = self.tag_service.generate_tags_and_embedding(description)
            
            # Convert tags to string for storage and display
            tag_string = self.tag_service.tag_generator.tags_to_string(tags)
            
            # Store in database
            self._add_firm(name, description, tags, tag_string, embedding)
            
            logger.debug(f"Row {row_num}: Added {name} with tags: {tag_string}")
            return "added"
            
        except (ValidationError, EmbeddingError) as e:
            logger.error(f"Row {row_num}: Tag generation failed for {name}: {e}")
            return "tag_error"
        except Exception as e:
            logger.error(f"Row {row_num}: Failed to process {name}: {e}")
            return "skipped"
    
    def _add_firm(self, name: str, description: str, tags: dict, tag_string: str, embedding: np.ndarray):
        """Add a single firm to database with tags and embedding"""
        # Insert into database
        self.cursor.execute('''
            INSERT INTO pe_firms (name, description, tags, tag_string) 
            VALUES (?, ?, ?, ?)
        ''', (name, description, json.dumps(tags), tag_string))
        
        firm_id = self.cursor.lastrowid
        self.conn.commit()
        
        # Store embedding for FAISS index
        self.embeddings.append(embedding)
        self.firm_ids.append(firm_id)
        
        logger.debug(f"Added: {name} (ID: {firm_id})")
    
    def _build_faiss_index(self):
        """Build FAISS index from tag embeddings"""
        if not self.embeddings:
            logger.warning("No embeddings to index")
            return
        
        logger.info("Building FAISS index from tag embeddings...")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings_array)
        
        # Save index to file
        index_file = self.db_path.replace('.db', '_index.faiss')
        faiss.write_index(self.faiss_index, index_file)
        
        # Save firm IDs mapping
        ids_file = self.db_path.replace('.db', '_ids.npy')
        np.save(ids_file, np.array(self.firm_ids))
        
        logger.info(f"FAISS index built with {len(self.embeddings)} tag embeddings")
        logger.info(f"Index saved to: {index_file}")
        logger.info(f"IDs mapping saved to: {ids_file}")
    
    def get_firm_tags(self, firm_id: int) -> dict:
        """Get tags for a specific firm"""
        self.cursor.execute('SELECT tags FROM pe_firms WHERE id = ?', (firm_id,))
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        return {}
    
    def get_all_unique_tags(self) -> dict:
        """Get all unique tags across all firms, organized by category"""
        self.cursor.execute('SELECT tags FROM pe_firms')
        rows = self.cursor.fetchall()
        
        all_tags = {
            "industry": set(),
            "technology": set(),
            "business_model": set(),
            "stage": set(),
            "market": set(),
            "solution_type": set()
        }
        
        for row in rows:
            tags = json.loads(row[0])
            for category, category_tags in tags.items():
                if category in all_tags:
                    all_tags[category].update(category_tags)
        
        # Convert sets to sorted lists
        return {category: sorted(list(tag_set)) for category, tag_set in all_tags.items()}


def main():
    if len(sys.argv) != 2:
        print("Usage: python load_pe_firms_tags.py <csv_file>")
        print("CSV format: company name, description")
        print("\nThis version generates structured tags from descriptions using ChatGPT")
        print("and creates embeddings from those tags for better similarity matching.")
        return 1
    
    csv_file = sys.argv[1]
    
    try:
        with PEFirmTagLoader() as loader:
            count = loader.load_from_csv(csv_file)
            
            print(f"\n✅ Successfully loaded {count} PE firms with tag-based embeddings!")
            print("\nFiles created:")
            print("  - pe_firms_tags.db (SQLite database with tags)")
            print("  - pe_firms_tags_index.faiss (FAISS vector index)")
            print("  - pe_firms_tags_ids.npy (ID mapping)")
            
            # Show some statistics
            if count > 0:
                unique_tags = loader.get_all_unique_tags()
                print(f"\n📊 Tag Statistics:")
                for category, tags in unique_tags.items():
                    print(f"  - {category}: {len(tags)} unique tags")
                    if tags:
                        print(f"    Examples: {', '.join(tags[:3])}")
            
            print("\n🔍 Ready for tag-based searching!")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())