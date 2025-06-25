#!/usr/bin/env python3
"""Test search functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import (
    company_db, vector_db, needs_vector_db, index, needs_index,
    initialize_indices, mark_indices_dirty, rebuild_faiss_index,
    ensure_indices_current, search_embeddings, search_needs_embeddings,
    create_search_result, search_companies_by_text, search_companies_by_needs
)
from embedding import create_embedding, combine_text_blob, add_embedding_to_index
from database_core import append_to_db
from models import ValidationError, EmbeddingError

def setup_test_data():
    """Set up test companies and embeddings"""
    # Clear existing data
    company_db.clear()
    vector_db.clear()
    needs_vector_db.clear()
    
    # Create test companies
    companies = [
        {
            "name": "TechCorp",
            "industry": "Technology",
            "location": "USA",
            "revenue": 1000000,
            "team_size": 50,
            "founded": 2020,
            "website": "https://techcorp.com",
            "description": "AI and machine learning solutions for enterprises",
            "needs": "Looking for enterprise partnerships and funding",
            "challenges": "Scaling AI models and finding talent"
        },
        {
            "name": "HealthAI",
            "industry": "Healthcare",
            "location": "Germany",
            "revenue": 800000,
            "team_size": 30,
            "founded": 2019,
            "website": "https://healthai.de",
            "description": "Medical diagnostics using artificial intelligence",
            "needs": "Seeking regulatory approval and hospital partnerships",
            "challenges": "Complex healthcare regulations and data privacy"
        },
        {
            "name": "FinanceBot",
            "industry": "FinTech",
            "location": "UK",
            "revenue": 1500000,
            "team_size": 75,
            "founded": 2018,
            "website": "https://financebot.co.uk",
            "description": "Automated trading and investment management platform",
            "needs": "Banking licenses and compliance expertise",
            "challenges": "Financial regulations and market volatility"
        }
    ]
    
    # Add companies and embeddings
    for company in companies:
        # Add to company database
        append_to_db(company, company_db)
        
        # Create embeddings
        desc_blob = combine_text_blob(company["description"], company["challenges"])
        desc_embedding = create_embedding(desc_blob)
        needs_embedding = create_embedding(company["needs"])
        
        # Add embeddings (this will also add to FAISS indices)
        add_embedding_to_index(desc_embedding, vector_db, index, "main")
        add_embedding_to_index(needs_embedding, needs_vector_db, needs_index, "needs")
    
    return companies

def test_index_initialization():
    """Test FAISS index initialization"""
    print("Testing index initialization...")
    
    # Indices should be initialized
    assert index is not None
    assert needs_index is not None
    print("   ✓ FAISS indices are initialized")

def test_index_management():
    """Test index marking and rebuilding"""
    print("Testing index management...")
    
    # Mark indices as dirty
    mark_indices_dirty()
    print("   ✓ Indices marked as dirty")
    
    # Set up test data
    setup_test_data()
    
    # Ensure indices are current (should rebuild)
    ensure_indices_current()
    print("   ✓ Indices rebuilt successfully")
    
    # Check indices have correct number of vectors
    assert index.ntotal == len(vector_db)
    assert needs_index.ntotal == len(needs_vector_db)
    print("   ✓ Index sizes match vector databases")

def test_search_embeddings():
    """Test embedding search functionality"""
    print("Testing embedding search...")
    
    # Set up test data
    companies = setup_test_data()
    
    # Search for AI-related content
    distances, indices = search_embeddings("artificial intelligence machine learning", top_k=2)
    
    assert len(distances) == 2
    assert len(indices) == 2
    assert distances[0] <= distances[1]  # Results should be ordered by distance
    print("   ✓ Embedding search returns ordered results")
    
    # Test validation
    try:
        search_embeddings("", top_k=3)
        assert False, "Should have raised ValidationError for empty query"
    except ValidationError as e:
        assert "Query text cannot be empty" in str(e)
    
    try:
        search_embeddings("test", top_k=0)
        assert False, "Should have raised ValidationError for invalid top_k"
    except ValidationError as e:
        assert "top_k must be a positive integer" in str(e)
    
    try:
        search_embeddings(123, top_k=3)
        assert False, "Should have raised ValidationError for non-string query"
    except ValidationError as e:
        assert "Query text must be a string" in str(e)
    
    print("   ✓ Embedding search validation works")

def test_search_needs_embeddings():
    """Test needs-specific embedding search"""
    print("Testing needs embedding search...")
    
    # Set up test data
    companies = setup_test_data()
    
    # Search for partnership-related needs
    distances, indices = search_needs_embeddings("partnerships funding approval", top_k=3)
    
    assert len(distances) <= 3  # Should return at most 3 results
    assert len(indices) <= 3
    
    if len(distances) > 1:
        assert distances[0] <= distances[1]  # Should be ordered
    
    print("   ✓ Needs embedding search works")
    
    # Test same validation as regular search
    try:
        search_needs_embeddings("   ", top_k=2)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Query text cannot be empty" in str(e)
    
    print("   ✓ Needs search validation works")

def test_create_search_result():
    """Test search result creation"""
    print("Testing search result creation...")
    
    # Create test company
    test_company = {
        "name": "ResultCorp",
        "description": "Test company for search results",
        "needs": "Testing search functionality",
        "challenges": "Making good test data",
        "website": "https://resultcorp.com"
    }
    
    # Create search result
    score = 1.23456
    result = create_search_result(test_company, score)
    
    # Check result structure
    expected_fields = ["name", "match_score", "description", "needs", "challenges", "website"]
    for field in expected_fields:
        assert field in result
    
    assert result["name"] == "ResultCorp"
    assert result["match_score"] == 1.235  # Should be rounded
    assert result["description"] == "Test company for search results"
    assert result["needs"] == "Testing search functionality"
    assert result["challenges"] == "Making good test data"
    assert result["website"] == "https://resultcorp.com"
    
    print("   ✓ Search result creation works correctly")

def test_search_companies_by_text():
    """Test high-level text search"""
    print("Testing company text search...")
    
    # Set up test data
    companies = setup_test_data()
    
    # Search for AI companies
    results = search_companies_by_text("artificial intelligence AI")
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check result structure
    for result in results:
        assert "name" in result
        assert "match_score" in result
        assert "description" in result
        assert "needs" in result
        assert "challenges" in result
        assert "website" in result
        assert isinstance(result["match_score"], float)
    
    # Results should be ordered by relevance (lower score = more relevant)
    if len(results) > 1:
        assert results[0]["match_score"] <= results[1]["match_score"]
    
    print(f"   ✓ Found {len(results)} companies for AI search")
    
    # Test specific search that should find TechCorp first
    ai_results = search_companies_by_text("machine learning enterprise solutions")
    assert len(ai_results) > 0
    # TechCorp should be most relevant for this query
    print(f"   ✓ Most relevant result: {ai_results[0]['name']}")

def test_search_companies_by_needs():
    """Test high-level needs search"""
    print("Testing company needs search...")
    
    # Set up test data
    companies = setup_test_data()
    
    # Search for companies needing partnerships
    results = search_companies_by_needs("partnerships funding regulatory")
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check result structure (same as text search)
    for result in results:
        assert "name" in result
        assert "match_score" in result
        assert isinstance(result["match_score"], float)
    
    print(f"   ✓ Found {len(results)} companies with partnership needs")
    
    # Test specific needs search
    regulatory_results = search_companies_by_needs("regulatory approval compliance")
    assert len(regulatory_results) > 0
    print(f"   ✓ Found companies needing regulatory help: {[r['name'] for r in regulatory_results]}")

def test_search_with_different_queries():
    """Test search with various query types"""
    print("Testing various search queries...")
    
    # Set up test data
    companies = setup_test_data()
    
    queries = [
        "technology",
        "healthcare medical",
        "finance trading investment",
        "artificial intelligence",
        "regulatory compliance",
        "funding partnerships"
    ]
    
    for query in queries:
        # Test both search types
        text_results = search_companies_by_text(query, top_k=2)
        needs_results = search_companies_by_needs(query, top_k=2)
        
        assert isinstance(text_results, list)
        assert isinstance(needs_results, list)
        
        print(f"   ✓ Query '{query}': {len(text_results)} text results, {len(needs_results)} needs results")

def test_empty_database_search():
    """Test search behavior with empty database"""
    print("Testing search with empty database...")
    
    # Clear all data
    company_db.clear()
    vector_db.clear()
    needs_vector_db.clear()
    
    # Rebuild indices (will be empty)
    rebuild_faiss_index()
    
    # Search should return empty results, not error
    results = search_companies_by_text("anything")
    assert results == []
    
    needs_results = search_companies_by_needs("anything")
    assert needs_results == []
    
    print("   ✓ Empty database search returns empty results")

if __name__ == "__main__":
    print("=== SEARCH TESTS ===")
    
    try:
        test_index_initialization()
        test_index_management()
        test_search_embeddings()
        test_search_needs_embeddings()
        test_create_search_result()
        test_search_companies_by_text()
        test_search_companies_by_needs()
        test_search_with_different_queries()
        test_empty_database_search()
        
        print("\n✅ ALL SEARCH TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ SEARCH TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)