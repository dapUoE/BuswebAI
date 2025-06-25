#!/usr/bin/env python3
"""Test embedding operations and FAISS management"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
from embedding import (
    encode_text, convert_to_numpy_array, create_embedding, 
    add_embedding_to_index, search_faiss_index, get_first_distances,
    get_first_indices, combine_text_blob, round_score, model, dimension
)
from models import ValidationError, EmbeddingError

def test_model_initialization():
    """Test that the model is properly initialized"""
    print("Testing model initialization...")
    
    # Model should be loaded
    assert model is not None
    print("   ✓ Model is initialized")
    
    # Dimension should be set
    assert isinstance(dimension, int)
    assert dimension > 0
    print(f"   ✓ Embedding dimension is {dimension}")

def test_encode_text():
    """Test text encoding to embeddings"""
    print("Testing text encoding...")
    
    # Test valid text
    text = "This is a test sentence"
    embedding = encode_text(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (dimension,)
    assert embedding.dtype == np.float32
    print("   ✓ Text encoding produces correct embedding")
    
    # Test different texts produce different embeddings
    text1 = "Technology startup"
    text2 = "Healthcare company"
    emb1 = encode_text(text1)
    emb2 = encode_text(text2)
    
    assert not np.array_equal(emb1, emb2)
    print("   ✓ Different texts produce different embeddings")
    
    # Test error cases
    try:
        encode_text(123)  # Not a string
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Text must be a string" in str(e)
    
    try:
        encode_text("")  # Empty string
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Text cannot be empty" in str(e)
    
    try:
        encode_text("   ")  # Whitespace only
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Text cannot be empty" in str(e)
    
    print("   ✓ Text encoding validation works")

def test_convert_to_numpy_array():
    """Test embedding conversion for FAISS"""
    print("Testing numpy array conversion...")
    
    # Create test embedding
    embedding = np.random.rand(dimension).astype(np.float32)
    
    # Convert for FAISS
    faiss_array = convert_to_numpy_array(embedding)
    
    assert isinstance(faiss_array, np.ndarray)
    assert faiss_array.shape == (1, dimension)
    assert np.array_equal(faiss_array[0], embedding)
    print("   ✓ Numpy array conversion works")
    
    # Test error case
    try:
        convert_to_numpy_array("not an array")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Embedding must be a numpy array" in str(e)
    
    print("   ✓ Conversion validation works")

def test_create_embedding():
    """Test create_embedding wrapper function"""
    print("Testing create_embedding...")
    
    text = "Test embedding creation"
    embedding = create_embedding(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (dimension,)
    print("   ✓ create_embedding works correctly")

def test_add_embedding_to_index():
    """Test adding embeddings to FAISS index"""
    print("Testing embedding addition to index...")
    
    # Create test data
    vector_list = []
    test_index = faiss.IndexFlatL2(dimension)
    
    # Create test embedding
    text = "Test company description"
    embedding = create_embedding(text)
    
    # Add to index
    index_pos = add_embedding_to_index(embedding, vector_list, test_index, "test")
    
    assert index_pos == 0
    assert len(vector_list) == 1
    assert np.array_equal(vector_list[0], embedding)
    assert test_index.ntotal == 1
    print("   ✓ Adding embedding to index works")
    
    # Add another embedding
    text2 = "Another test description"
    embedding2 = create_embedding(text2)
    index_pos2 = add_embedding_to_index(embedding2, vector_list, test_index, "test")
    
    assert index_pos2 == 1
    assert len(vector_list) == 2
    assert test_index.ntotal == 2
    print("   ✓ Multiple embeddings can be added")
    
    # Test validation
    try:
        add_embedding_to_index("not an array", vector_list, test_index, "test")
        assert False, "Should have raised EmbeddingError"
    except EmbeddingError as e:
        assert "Embedding must be a numpy array" in str(e)
    
    # Test wrong shape
    try:
        wrong_shape = np.random.rand(dimension + 1).astype(np.float32)
        add_embedding_to_index(wrong_shape, vector_list, test_index, "test")
        assert False, "Should have raised EmbeddingError"
    except EmbeddingError as e:
        assert f"Embedding must have shape ({dimension},)" in str(e)
    
    print("   ✓ Embedding addition validation works")

def test_search_faiss_index():
    """Test FAISS index searching"""
    print("Testing FAISS index search...")
    
    # Create test index with embeddings
    vector_list = []
    test_index = faiss.IndexFlatL2(dimension)
    
    # Add test embeddings
    texts = [
        "Technology startup focused on AI",
        "Healthcare company using machine learning", 
        "Financial services with blockchain technology"
    ]
    
    for text in texts:
        embedding = create_embedding(text)
        add_embedding_to_index(embedding, vector_list, test_index, "test")
    
    # Search for similar content
    query_text = "AI and machine learning company"
    query_embedding = create_embedding(query_text)
    query_array = convert_to_numpy_array(query_embedding)
    
    # Search
    search_result = search_faiss_index(query_array, 2, test_index)
    
    assert isinstance(search_result, tuple)
    assert len(search_result) == 2  # distances and indices
    
    distances, indices = search_result
    assert distances.shape == (1, 2)  # 1 query, top 2 results
    assert indices.shape == (1, 2)
    
    # Results should be ordered by distance (smaller = more similar)
    assert distances[0, 0] <= distances[0, 1]
    print("   ✓ FAISS search works correctly")
    
    # Test validation
    try:
        search_faiss_index("not an array", 3, test_index)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Query array must be a numpy array" in str(e)
    
    try:
        search_faiss_index(query_array, 0, test_index)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "k must be a positive integer" in str(e)
    
    print("   ✓ Search validation works")

def test_search_result_extraction():
    """Test extracting distances and indices from search results"""
    print("Testing search result extraction...")
    
    # Create mock search result
    distances = np.array([[0.5, 1.2, 2.1]], dtype=np.float32)
    indices = np.array([[0, 2, 1]], dtype=np.int64)
    search_result = (distances, indices)
    
    # Test extraction functions
    first_distances = get_first_distances(search_result)
    first_indices = get_first_indices(search_result)
    
    assert np.array_equal(first_distances, distances[0])
    assert np.array_equal(first_indices, indices[0])
    print("   ✓ Search result extraction works")
    
    # Test validation
    try:
        get_first_distances("not a tuple")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Search result must be a tuple of length 2" in str(e)
    
    try:
        get_first_indices((distances,))  # Wrong length
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Search result must be a tuple of length 2" in str(e)
    
    print("   ✓ Result extraction validation works")

def test_combine_text_blob():
    """Test text combination for embeddings"""
    print("Testing text combination...")
    
    description = "AI-powered health monitoring"
    challenges = "Regulatory approval and data privacy"
    
    combined = combine_text_blob(description, challenges)
    expected = f"{description}. Challenges: {challenges}"
    
    assert combined == expected
    print("   ✓ Text combination works correctly")
    
    # Test validation
    try:
        combine_text_blob(123, "challenges")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "All text fields must be strings" in str(e)
    
    try:
        combine_text_blob("description", None)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "All text fields must be strings" in str(e)
    
    print("   ✓ Text combination validation works")

def test_round_score():
    """Test score rounding"""
    print("Testing score rounding...")
    
    # Test basic rounding
    assert round_score(1.23456) == 1.235  # Default 3 decimals
    assert round_score(1.23456, 2) == 1.23
    assert round_score(1.23456, 4) == 1.2346
    assert round_score(1.0) == 1.0
    print("   ✓ Score rounding works correctly")
    
    # Test validation
    try:
        round_score("not a number")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Score must be a number" in str(e)
    
    try:
        round_score(1.5, -1)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Decimals must be a non-negative integer" in str(e)
    
    try:
        round_score(1.5, "not an int")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Decimals must be a non-negative integer" in str(e)
    
    print("   ✓ Score rounding validation works")

def test_embedding_similarity():
    """Test that similar texts produce similar embeddings"""
    print("Testing embedding similarity...")
    
    # Similar texts
    text1 = "AI startup developing machine learning solutions"
    text2 = "Machine learning company building AI products"
    
    # Different text
    text3 = "Traditional manufacturing company making steel products"
    
    emb1 = create_embedding(text1)
    emb2 = create_embedding(text2)
    emb3 = create_embedding(text3)
    
    # Calculate cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_12 = cosine_similarity(emb1, emb2)  # Similar texts
    sim_13 = cosine_similarity(emb1, emb3)  # Different texts
    
    # Similar texts should be more similar than different texts
    assert sim_12 > sim_13
    print(f"   ✓ Similar texts more similar ({sim_12:.3f}) than different texts ({sim_13:.3f})")

if __name__ == "__main__":
    print("=== EMBEDDING TESTS ===")
    
    try:
        test_model_initialization()
        test_encode_text()
        test_convert_to_numpy_array()
        test_create_embedding()
        test_add_embedding_to_index()
        test_search_faiss_index()
        test_search_result_extraction()
        test_combine_text_blob()
        test_round_score()
        test_embedding_similarity()
        
        print("\n✅ ALL EMBEDDING TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ EMBEDDING TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)