"""
Tag-based embedding service that generates tags from descriptions and creates embeddings.
"""

import json
import numpy as np
from typing import Dict, List, Optional
from openai import OpenAI
import os

from tag_generator import TagGenerator
from config import DatabaseConfig
from models import ValidationError, EmbeddingError


class TagEmbeddingService:
    """Service for generating tags and creating embeddings from those tags."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the tag embedding service."""
        self.tag_generator = TagGenerator()
        self.embedding_model = model_name or DatabaseConfig.OPENAI_EMBEDDING_MODEL
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
    
    def generate_tags_and_embedding(self, description: str) -> tuple[Dict[str, List[str]], np.ndarray]:
        """
        Generate tags from description and create embedding from those tags.
        
        Args:
            description: Company description text
            
        Returns:
            Tuple of (tags_dict, embedding_array)
        """
        if not isinstance(description, str):
            raise ValidationError("Description must be a string")
        
        if not description.strip():
            raise ValidationError("Description cannot be empty")
        
        try:
            # Step 1: Generate tags using ChatGPT
            tags = self.tag_generator.generate_tags(description)
            
            # Step 2: Convert tags to embedding
            embedding = self.create_embedding_from_tags(tags)
            
            return tags, embedding
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate tags and embedding: {str(e)}")
    
    def create_embedding_from_tags(self, tags: Dict[str, List[str]]) -> np.ndarray:
        """
        Create embedding from tags dictionary.
        
        Args:
            tags: Dictionary of categorized tags
            
        Returns:
            Numpy array embedding
        """
        try:
            # Convert tags to string representation
            tag_string = self.tag_generator.tags_to_string(tags)
            
            # If no tags, create a default representation
            if not tag_string.strip():
                tag_string = "unknown:general"
            
            # Create embedding using OpenAI
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=tag_string
            )
            
            return np.array(response.data[0].embedding)
            
        except Exception as e:
            raise EmbeddingError(f"Failed to create embedding from tags: {str(e)}")
    
    def create_embedding_from_tag_string(self, tag_string: str) -> np.ndarray:
        """
        Create embedding directly from a tag string.
        
        Args:
            tag_string: Space-separated tag string
            
        Returns:
            Numpy array embedding
        """
        if not isinstance(tag_string, str):
            raise ValidationError("Tag string must be a string")
        
        if not tag_string.strip():
            tag_string = "unknown:general"
        
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=tag_string
            )
            return np.array(response.data[0].embedding)
            
        except Exception as e:
            raise EmbeddingError(f"Failed to create embedding from tag string: {str(e)}")
    
    def calculate_tag_similarity(self, tags1: Dict[str, List[str]], tags2: Dict[str, List[str]]) -> float:
        """
        Calculate similarity between two tag sets using embeddings.
        
        Args:
            tags1: First tag set
            tags2: Second tag set
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            emb1 = self.create_embedding_from_tags(tags1)
            emb2 = self.create_embedding_from_tags(tags2)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
            
        except Exception as e:
            raise EmbeddingError(f"Failed to calculate tag similarity: {str(e)}")
    
    def get_tag_categories(self) -> Dict[str, List[str]]:
        """Get available tag categories and their examples."""
        return self.tag_generator.tag_categories.copy()
    
    def validate_tags(self, tags: Dict[str, List[str]]) -> bool:
        """
        Validate that tags follow the expected structure.
        
        Args:
            tags: Tags dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(tags, dict):
            return False
        
        expected_categories = set(self.tag_generator.tag_categories.keys())
        tag_categories = set(tags.keys())
        
        # Check if all expected categories are present
        if not expected_categories.issubset(tag_categories):
            return False
        
        # Check that all values are lists of strings
        for category, category_tags in tags.items():
            if not isinstance(category_tags, list):
                return False
            if not all(isinstance(tag, str) for tag in category_tags):
                return False
        
        return True


# Convenience functions
def generate_company_tags_and_embedding(description: str) -> tuple[Dict[str, List[str]], np.ndarray]:
    """Generate tags and embedding for a company description."""
    service = TagEmbeddingService()
    return service.generate_tags_and_embedding(description)


def create_embedding_from_tags(tags: Dict[str, List[str]]) -> np.ndarray:
    """Create embedding from tags dictionary."""
    service = TagEmbeddingService()
    return service.create_embedding_from_tags(tags)


# Example usage and testing
if __name__ == "__main__":
    # Test the tag embedding service
    test_descriptions = [
        "AI-powered healthcare platform that uses machine learning to analyze medical images and assist radiologists in early disease detection",
        "B2B SaaS fintech company providing automated accounting and financial management tools for small businesses"
    ]
    
    service = TagEmbeddingService()
    
    for i, desc in enumerate(test_descriptions, 1):
        print(f"\n=== Test {i} ===")
        print(f"Description: {desc}")
        
        try:
            tags, embedding = service.generate_tags_and_embedding(desc)
            print(f"Generated tags: {json.dumps(tags, indent=2)}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding sample: {embedding[:5]}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Test similarity between the two descriptions
    if len(test_descriptions) >= 2:
        try:
            tags1, _ = service.generate_tags_and_embedding(test_descriptions[0])
            tags2, _ = service.generate_tags_and_embedding(test_descriptions[1])
            
            similarity = service.calculate_tag_similarity(tags1, tags2)
            print(f"\nSimilarity between companies: {similarity:.3f}")
            
        except Exception as e:
            print(f"Similarity calculation error: {e}")