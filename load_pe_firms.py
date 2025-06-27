#!/usr/bin/env python3
"""
Load PE firms from CSV file into database with vector embeddings.

Usage: python load_pe_firms.py pe_firms.csv

CSV format: company name, description
"""

import sys
import os
import csv
import sqlite3
import numpy as np
import faiss
from openai import OpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class PEFirmLoader:
    def __init__(self, db_path='pe_firms.db'):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.client = None
        self.embeddings = []
        self.firm_ids = []
        self.faiss_index = None
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self._init_database()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def _init_database(self):
        """Initialize database tables"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS pe_firms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL
            )
        ''')
        self.conn.commit()
        logger.info("Database initialized")
    
    def _load_client(self):
        """Load OpenAI client"""
        if self.client is None:
            logger.info("Loading OpenAI client...")
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            logger.info("OpenAI client loaded successfully")
    
    def _create_embedding(self, text):
        """Create embedding for text"""
        self._load_client()
        response = self.client.embeddings.create(
            model='text-embedding-3-small',
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def load_from_csv(self, csv_file):
        """Load PE firms from CSV file"""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        logger.info(f"Loading PE firms from {csv_file}")
        
        # Clear existing data
        self.cursor.execute("DELETE FROM pe_firms")
        self.conn.commit()
        
        added_count = 0
        skipped_count = 0
        
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
                    name = first_row[0].strip()
                    description = first_row[1].strip()
                    if name and description:
                        self._add_firm(name, description)
                        added_count += 1
                    else:
                        skipped_count += 1
            
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
                
                try:
                    self._add_firm(name, description)
                    added_count += 1
                    
                    if added_count % 10 == 0:
                        logger.info(f"Processed {added_count} firms...")
                        
                except Exception as e:
                    logger.error(f"Row {row_num}: Failed to add {name}: {e}")
                    skipped_count += 1
        
        logger.info(f"Completed: {added_count} firms added, {skipped_count} skipped")
        
        # Build FAISS index
        self._build_faiss_index()
        
        return added_count
    
    def _add_firm(self, name, description):
        """Add a single firm to database"""
        # Insert into database
        self.cursor.execute('''
            INSERT INTO pe_firms (name, description) VALUES (?, ?)
        ''', (name, description))
        
        firm_id = self.cursor.lastrowid
        self.conn.commit()
        
        # Create and store embedding
        embedding = self._create_embedding(description)
        self.embeddings.append(embedding)
        self.firm_ids.append(firm_id)
        
        logger.debug(f"Added: {name} (ID: {firm_id})")
    
    def _build_faiss_index(self):
        """Build FAISS index from embeddings"""
        if not self.embeddings:
            logger.warning("No embeddings to index")
            return
        
        logger.info("Building FAISS index...")
        
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
        
        logger.info(f"FAISS index built with {len(self.embeddings)} embeddings")
        logger.info(f"Index saved to: {index_file}")
        logger.info(f"IDs mapping saved to: {ids_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python load_pe_firms.py <csv_file>")
        print("CSV format: company name, description")
        return 1
    
    csv_file = sys.argv[1]
    
    try:
        with PEFirmLoader() as loader:
            count = loader.load_from_csv(csv_file)
            print(f"\nSuccessfully loaded {count} PE firms!")
            print("Files created:")
            print("  - pe_firms.db (SQLite database)")
            print("  - pe_firms_index.faiss (FAISS vector index)")
            print("  - pe_firms_ids.npy (ID mapping)")
            print("\nReady for searching!")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())