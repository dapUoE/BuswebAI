#!/usr/bin/env python3
"""Test EmbeddingService class"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
from embedding_service import EmbeddingService
from models import ValidationError, EmbeddingError

def test_embedding_service_initialization():
    """Test EmbeddingService initialization"""
    print("Testing EmbeddingService initialization...")
    
    # Test default initialization
    service = EmbeddingService()
    assert service._model_name == 'all-MiniLM-L6-v2'
    
    # Test custom model name
    custom_service = EmbeddingService("custom-model")
    assert custom_service._model_name == "custom-model"
    
    print("   ✓ EmbeddingService initializes correctly")

def test_lazy_loading():
    """Test lazy loading of model"""
    print("Testing lazy loading...")
    
    service = EmbeddingService()
    
    # Model should not be loaded initially
    assert service._model is None
    assert service._dimension is None
    
    # Access model property to trigger loading
    model = service.model
    assert model is not None
    assert service._model is not None
    
    # Access dimension property
    dimension = service.dimension
    assert isinstance(dimension, int)
    assert dimension > 0
    assert service._dimension == dimension
    
    print(f"   ✓ Lazy loading works, dimension: {dimension}")

def test_text_embedding_creation():
    """Test text embedding creation"""
    print("Testing text embedding creation...")
    
    service = EmbeddingService()
    
    # Test valid text
    text = "This is a test sentence for embedding"
    embedding = service.create_text_embedding(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (service.dimension,)
    assert embedding.dtype == np.float32
    
    print("   ✓ Text embedding creation works")
    
    # Test different texts produce different embeddings
    text1 = "Technology startup"
    text2 = "Healthcare company"
    emb1 = service.create_text_embedding(text1)
    emb2 = service.create_text_embedding(text2)
    
    assert not np.array_equal(emb1, emb2)
    print("   ✓ Different texts produce different embeddings")

def test_specialized_embedding_methods():
    """Test specialized embedding creation methods"""
    print("Testing specialized embedding methods...")
    
    service = EmbeddingService()
    
    # Test description embedding
    description = "AI-powered healthcare solutions"
    challenges = "Regulatory compliance and data privacy"
    desc_embedding = service.create_description_embedding(description, challenges)
    
    assert isinstance(desc_embedding, np.ndarray)
    assert desc_embedding.shape == (service.dimension,)
    print("   ✓ Description embedding creation works")
    
    # Test needs embedding
    needs = "Looking for hospital partnerships and funding"
    needs_embedding = service.create_needs_embedding(needs)
    
    assert isinstance(needs_embedding, np.ndarray)
    assert needs_embedding.shape == (service.dimension,)
    print("   ✓ Needs embedding creation works")

def test_similarity_calculation():
    """Test text similarity calculation"""
    print("Testing similarity calculation...")
    
    service = EmbeddingService()
    
    # Similar texts
    text1 = "AI startup developing machine learning solutions"
    text2 = "Machine learning company building AI products"
    
    # Different text
    text3 = "Traditional manufacturing steel production"
    
    sim_12 = service.calculate_similarity(text1, text2)
    sim_13 = service.calculate_similarity(text1, text3)
    
    assert isinstance(sim_12, float)
    assert isinstance(sim_13, float)
    assert -1 <= sim_12 <= 1
    assert -1 <= sim_13 <= 1
    
    # Similar texts should be more similar
    assert sim_12 > sim_13
    
    print(f"   ✓ Similarity calculation works (similar: {sim_12:.3f}, different: {sim_13:.3f})")

def test_index_search():
    """Test FAISS index search functionality"""
    print("Testing index search...")
    
    service = EmbeddingService()
    
    # Create test index
    test_index = faiss.IndexFlatL2(service.dimension)
    
    # Add test embeddings
    texts = [
        "Technology startup focused on AI",
        "Healthcare company using machine learning",
        "Financial services with blockchain"
    ]
    
    for text in texts:
        embedding = service.create_text_embedding(text)
        from embedding import convert_to_numpy_array
        embedding_array = convert_to_numpy_array(embedding)
        test_index.add(embedding_array)
    
    # Search
    query = "AI and machine learning company"
    distances, indices = service.search_index(query, test_index, top_k=2)
    
    assert len(distances) == 2
    assert len(indices) == 2
    assert distances[0] <= distances[1]  # Results should be ordered
    
    print("   ✓ Index search works correctly")

def test_score_rounding():
    """Test score rounding"""
    print("Testing score rounding...")
    
    service = EmbeddingService()
    
    # Test basic rounding
    assert service.round_score(1.23456) == 1.235  # Default 3 decimals
    assert service.round_score(1.23456, 2) == 1.23
    assert service.round_score(1.23456, 4) == 1.2346
    
    print("   ✓ Score rounding works")

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    service = EmbeddingService()
    
    # Test empty text embedding
    try:
        service.create_text_embedding("")
        assert False, "Should have raised ValidationError"
    except EmbeddingError:
        pass
    
    # Test invalid search parameters
    test_index = faiss.IndexFlatL2(service.dimension)
    
    try:
        service.search_index("", test_index)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    try:
        service.search_index("test", test_index, top_k=0)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    try:
        service.search_index(123, test_index)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    print("   ✓ Error handling works")

def test_multiple_service_instances():
    """Test multiple service instances"""
    print("Testing multiple service instances...")
    
    service1 = EmbeddingService()
    service2 = EmbeddingService()
    
    # Both should work independently
    emb1 = service1.create_text_embedding("test text 1")
    emb2 = service2.create_text_embedding("test text 1")
    
    # Should produce same embeddings for same text
    assert np.array_equal(emb1, emb2)
    
    # Should have same dimensions
    assert service1.dimension == service2.dimension
    
    print("   ✓ Multiple service instances work correctly")

if __name__ == "__main__":
    print("=== EMBEDDING SERVICE TESTS ===")
    
    try:
        test_embedding_service_initialization()
        test_lazy_loading()
        test_text_embedding_creation()
        test_specialized_embedding_methods()
        test_similarity_calculation()
        test_index_search()
        test_score_rounding()
        test_error_handling()
        test_multiple_service_instances()
        
        print("\n✅ ALL EMBEDDING SERVICE TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ EMBEDDING SERVICE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)