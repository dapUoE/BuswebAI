#!/usr/bin/env python3
"""Test configuration constants and settings"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DatabaseConfig

def test_model_configuration():
    """Test model configuration constants"""
    print("Testing model configuration...")
    
    # Test OpenAI embedding model is set
    assert DatabaseConfig.OPENAI_EMBEDDING_MODEL == 'text-embedding-3-small'
    assert isinstance(DatabaseConfig.OPENAI_EMBEDDING_MODEL, str)
    assert len(DatabaseConfig.OPENAI_EMBEDDING_MODEL) > 0
    print("   ✓ OpenAI embedding model is valid")
    
    # Test embedding dimension is set
    assert DatabaseConfig.EMBEDDING_DIMENSION == 1536
    assert isinstance(DatabaseConfig.EMBEDDING_DIMENSION, int)
    assert DatabaseConfig.EMBEDDING_DIMENSION > 0
    print("   ✓ Embedding dimension is valid")

def test_search_configuration():
    """Test search configuration constants"""
    print("Testing search configuration...")
    
    # Test default top_k
    assert DatabaseConfig.DEFAULT_TOP_K == 3
    assert isinstance(DatabaseConfig.DEFAULT_TOP_K, int)
    assert DatabaseConfig.DEFAULT_TOP_K > 0
    print("   ✓ Default top_k is valid")
    
    # Test score decimals
    assert DatabaseConfig.DEFAULT_SCORE_DECIMALS == 3
    assert isinstance(DatabaseConfig.DEFAULT_SCORE_DECIMALS, int)
    assert DatabaseConfig.DEFAULT_SCORE_DECIMALS >= 0
    print("   ✓ Default score decimals is valid")

def test_required_fields():
    """Test required fields configuration"""
    print("Testing required fields...")
    
    expected_fields = [
        'name', 'industry', 'location', 'revenue', 'team_size', 
        'founded', 'website', 'description', 'needs', 'challenges'
    ]
    
    assert DatabaseConfig.REQUIRED_FIELDS == expected_fields
    assert isinstance(DatabaseConfig.REQUIRED_FIELDS, list)
    assert len(DatabaseConfig.REQUIRED_FIELDS) == 10
    
    # All fields should be strings
    for field in DatabaseConfig.REQUIRED_FIELDS:
        assert isinstance(field, str)
        assert len(field) > 0
    
    print("   ✓ Required fields are valid")

def test_validation_constants():
    """Test validation range constants"""
    print("Testing validation constants...")
    
    # Revenue range
    assert DatabaseConfig.MIN_REVENUE == 0
    assert DatabaseConfig.MAX_REVENUE == 1_000_000_000_000
    assert DatabaseConfig.MIN_REVENUE < DatabaseConfig.MAX_REVENUE
    print("   ✓ Revenue range is valid")
    
    # Team size range  
    assert DatabaseConfig.MIN_TEAM_SIZE == 1
    assert DatabaseConfig.MAX_TEAM_SIZE == 1_000_000
    assert DatabaseConfig.MIN_TEAM_SIZE < DatabaseConfig.MAX_TEAM_SIZE
    print("   ✓ Team size range is valid")
    
    # Founded year range
    assert DatabaseConfig.MIN_FOUNDED_YEAR == 1800
    assert DatabaseConfig.MAX_FOUNDED_YEAR == 2100
    assert DatabaseConfig.MIN_FOUNDED_YEAR < DatabaseConfig.MAX_FOUNDED_YEAR
    print("   ✓ Founded year range is valid")
    
    # String length
    assert DatabaseConfig.MAX_STRING_LENGTH == 10000
    assert isinstance(DatabaseConfig.MAX_STRING_LENGTH, int)
    assert DatabaseConfig.MAX_STRING_LENGTH > 0
    print("   ✓ Max string length is valid")

def test_constant_types():
    """Test that all constants have correct types"""
    print("Testing constant types...")
    
    # All numeric constants should be integers
    numeric_constants = [
        'DEFAULT_TOP_K', 'DEFAULT_SCORE_DECIMALS', 'MIN_REVENUE', 'MAX_REVENUE',
        'MIN_TEAM_SIZE', 'MAX_TEAM_SIZE', 'MIN_FOUNDED_YEAR', 'MAX_FOUNDED_YEAR',
        'MAX_STRING_LENGTH'
    ]
    
    for const_name in numeric_constants:
        const_value = getattr(DatabaseConfig, const_name)
        assert isinstance(const_value, int), f"{const_name} should be int, got {type(const_value)}"
    
    print("   ✓ All constant types are correct")

if __name__ == "__main__":
    print("=== CONFIG TESTS ===")
    
    try:
        test_model_configuration()
        test_search_configuration() 
        test_required_fields()
        test_validation_constants()
        test_constant_types()
        
        print("\n✅ ALL CONFIG TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ CONFIG TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)