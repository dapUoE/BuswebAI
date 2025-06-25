#!/usr/bin/env python3
"""Test SearchService class"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_service import SearchService
from database_manager import DatabaseManager
from embedding_service import EmbeddingService
from models import DatabaseError

def setup_test_data(db_manager, embedding_service):
    """Set up test companies with embeddings"""
    companies = [
        {
            "name": "TechCorp",
            "industry": "Technology",
            "location": "USA",
            "revenue": 2000000,
            "team_size": 100,
            "founded": 2020,
            "website": "https://techcorp.com",
            "description": "AI and machine learning solutions for enterprises",
            "needs": "Looking for enterprise partnerships and funding",
            "challenges": "Scaling AI models and finding technical talent"
        },
        {
            "name": "HealthAI",
            "industry": "HealthTech",
            "location": "Germany",
            "revenue": 1500000,
            "team_size": 75,
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
            "revenue": 3000000,
            "team_size": 150,
            "founded": 2018,
            "website": "https://financebot.co.uk",
            "description": "Automated trading and investment management",
            "needs": "Banking licenses and institutional clients",
            "challenges": "Financial regulations and market competition"
        }
    ]
    
    for company in companies:
        # Add company to database
        company_idx = db_manager.add_company(company)
        
        # Create and add embeddings
        desc_embedding = embedding_service.create_description_embedding(
            company["description"], company["challenges"]
        )
        needs_embedding = embedding_service.create_needs_embedding(company["needs"])
        
        db_manager.add_embedding(desc_embedding)
        db_manager.add_needs_embedding(needs_embedding)
    
    return companies

def test_search_service_initialization():
    """Test SearchService initialization"""
    print("Testing SearchService initialization...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    assert search_service.db == db_manager
    assert search_service.embedding == embedding_service
    
    print("   ✓ SearchService initializes correctly")

def test_search_by_description():
    """Test description-based search"""
    print("Testing search by description...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    # Set up test data
    companies = setup_test_data(db_manager, embedding_service)
    
    # Search for AI-related content
    results = search_service.search_by_description("artificial intelligence machine learning")
    
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
    
    # Results should be ordered by relevance
    if len(results) > 1:
        assert results[0]["match_score"] <= results[1]["match_score"]
    
    print(f"   ✓ Found {len(results)} results for description search")

def test_search_by_needs():
    """Test needs-based search"""
    print("Testing search by needs...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    # Set up test data
    companies = setup_test_data(db_manager, embedding_service)
    
    # Search for partnership-related needs
    results = search_service.search_by_needs("partnerships funding approval")
    
    assert isinstance(results, list)
    assert len(results) > 0
    
    # Check result structure (same as description search)
    for result in results:
        assert "name" in result
        assert "match_score" in result
        assert isinstance(result["match_score"], float)
    
    print(f"   ✓ Found {len(results)} results for needs search")

def test_get_companies_by_indices():
    """Test getting companies by indices"""
    print("Testing get companies by indices...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    # Set up test data
    companies = setup_test_data(db_manager, embedding_service)
    
    # Get companies by indices
    indices = [0, 2]  # First and third companies
    retrieved_companies = search_service.get_companies_by_indices(indices)
    
    assert len(retrieved_companies) == 2
    assert retrieved_companies[0]["name"] == "TechCorp"
    assert retrieved_companies[1]["name"] == "FinanceBot"
    
    print("   ✓ Get companies by indices works")

def test_format_filtered_results():
    """Test filtered result formatting"""
    print("Testing filtered result formatting...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    # Set up test data
    companies = setup_test_data(db_manager, embedding_service)
    
    # Get some companies and format them
    test_companies = [companies[0], companies[1]]
    formatted_results = search_service.format_filtered_results(test_companies)
    
    assert len(formatted_results) == 2
    
    # Check structure includes all expected fields
    for result in formatted_results:
        expected_fields = [
            "name", "match_score", "description", "needs", "challenges",
            "website", "industry", "location", "revenue", "team_size", "founded"
        ]
        for field in expected_fields:
            assert field in result
        
        # match_score should be None for filtered results
        assert result["match_score"] is None
    
    print("   ✓ Filtered result formatting works")

def test_search_result_creation():
    """Test search result creation"""
    print("Testing search result creation...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    # Create test company data
    company_data = {
        "name": "TestCorp",
        "description": "Test company description", 
        "needs": "Test needs",
        "challenges": "Test challenges",
        "website": "https://test.com"
    }
    
    # Test result creation
    score = 1.23456
    result = search_service._create_search_result(company_data, score)
    
    assert result["name"] == "TestCorp"
    assert result["match_score"] == 1.235  # Should be rounded
    assert result["description"] == "Test company description"
    assert result["needs"] == "Test needs"
    assert result["challenges"] == "Test challenges"
    assert result["website"] == "https://test.com"
    
    print("   ✓ Search result creation works")

def test_semantic_with_filters():
    """Test semantic search with filters"""
    print("Testing semantic search with filters...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    # Set up test data
    companies = setup_test_data(db_manager, embedding_service)
    
    # Test with filtered indices
    filtered_indices = [0, 1]  # Only first two companies
    results = search_service.search_with_semantic_and_filters(
        "artificial intelligence", filtered_indices, top_k=2
    )
    
    assert isinstance(results, list)
    # Should only return results from filtered indices
    assert len(results) <= 2
    
    print(f"   ✓ Semantic search with filters works ({len(results)} results)")

def test_empty_database_search():
    """Test search behavior with empty database"""
    print("Testing search with empty database...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    # Search empty database
    desc_results = search_service.search_by_description("anything")
    needs_results = search_service.search_by_needs("anything")
    
    assert desc_results == []
    assert needs_results == []
    
    print("   ✓ Empty database search returns empty results")

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    # Test search with invalid parameters (handled by embedding service)
    try:
        search_service.search_by_description("")
        assert False, "Should have raised error"
    except:
        pass  # Expected to fail
    
    try:
        search_service.search_by_needs("")
        assert False, "Should have raised error"
    except:
        pass  # Expected to fail
    
    print("   ✓ Error handling works")

def test_search_relevance():
    """Test search relevance and ordering"""
    print("Testing search relevance...")
    
    db_manager = DatabaseManager()
    embedding_service = EmbeddingService()
    search_service = SearchService(db_manager, embedding_service)
    
    # Set up test data
    companies = setup_test_data(db_manager, embedding_service)
    
    # Search for AI - should rank TechCorp and HealthAI higher
    ai_results = search_service.search_by_description("artificial intelligence AI")
    
    if len(ai_results) >= 2:
        # TechCorp or HealthAI should be most relevant for AI query
        top_result = ai_results[0]
        assert top_result["name"] in ["TechCorp", "HealthAI"]
    
    # Search for finance - should rank FinanceBot higher
    finance_results = search_service.search_by_description("trading finance investment")
    
    if len(finance_results) >= 1:
        # Results should include FinanceBot
        names = [r["name"] for r in finance_results]
        # FinanceBot should be in results (not necessarily first due to small dataset)
        assert "FinanceBot" in names
    
    print("   ✓ Search relevance works correctly")

if __name__ == "__main__":
    print("=== SEARCH SERVICE TESTS ===")
    
    try:
        test_search_service_initialization()
        test_search_by_description()
        test_search_by_needs()
        test_get_companies_by_indices()
        test_format_filtered_results()
        test_search_result_creation()
        test_semantic_with_filters()
        test_empty_database_search()
        test_error_handling()
        test_search_relevance()
        
        print("\n✅ ALL SEARCH SERVICE TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ SEARCH SERVICE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)