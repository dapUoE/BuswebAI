#!/usr/bin/env python3
"""Test DatabaseManager class"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from database_manager import DatabaseManager
from models import ValidationError, DatabaseError, CompanyNotFoundError

def test_database_manager_initialization():
    """Test DatabaseManager initialization"""
    print("Testing DatabaseManager initialization...")
    
    db = DatabaseManager()
    
    # Check initial state
    assert db.get_company_count() == 0
    assert db.get_vector_count() == 0
    assert db.get_needs_vector_count() == 0
    assert db.get_all_indices() == []
    
    # Check indices are initialized
    main_index = db.get_main_index()
    needs_index = db.get_needs_index()
    assert main_index is not None
    assert needs_index is not None
    
    print("   ✓ DatabaseManager initializes correctly")

def test_company_crud_operations():
    """Test company CRUD operations"""
    print("Testing company CRUD operations...")
    
    db = DatabaseManager()
    
    # Test data
    company_data = {
        "name": "TestCorp",
        "industry": "Technology",
        "location": "USA",
        "revenue": 1000000,
        "team_size": 50,
        "founded": 2020,
        "website": "https://testcorp.com",
        "description": "Test company",
        "needs": "Testing needs",
        "challenges": "Testing challenges"
    }
    
    # CREATE
    company_idx = db.add_company(company_data)
    assert company_idx == 0
    assert db.get_company_count() == 1
    print("   ✓ Company creation works")
    
    # READ
    retrieved = db.get_company(0)
    assert retrieved["name"] == "TestCorp"
    assert retrieved["revenue"] == 1000000
    print("   ✓ Company retrieval works")
    
    # UPDATE
    updated_data = company_data.copy()
    updated_data["name"] = "UpdatedCorp"
    updated_data["revenue"] = 2000000
    
    result = db.update_company(0, updated_data)
    assert result == True
    
    updated = db.get_company(0)
    assert updated["name"] == "UpdatedCorp"
    assert updated["revenue"] == 2000000
    print("   ✓ Company update works")
    
    # DELETE
    result = db.delete_company(0)
    assert result == True
    assert db.get_company_count() == 0
    print("   ✓ Company deletion works")

def test_vector_operations():
    """Test vector database operations"""
    print("Testing vector operations...")
    
    db = DatabaseManager()
    
    # Create test embeddings
    embedding1 = np.random.rand(384).astype(np.float32)
    embedding2 = np.random.rand(384).astype(np.float32)
    
    # Add embeddings
    idx1 = db.add_embedding(embedding1)
    idx2 = db.add_needs_embedding(embedding2)
    
    assert idx1 == 0
    assert idx2 == 0
    assert db.get_vector_count() == 1
    assert db.get_needs_vector_count() == 1
    
    print("   ✓ Vector addition works")
    
    # Test embedding updates
    new_embedding1 = np.random.rand(384).astype(np.float32)
    new_embedding2 = np.random.rand(384).astype(np.float32)
    
    db.update_embeddings(0, new_embedding1, new_embedding2)
    assert db.get_vector_count() == 1
    assert db.get_needs_vector_count() == 1
    
    print("   ✓ Vector updates work")

def test_index_management():
    """Test FAISS index management"""
    print("Testing index management...")
    
    db = DatabaseManager()
    
    # Add some embeddings
    for i in range(3):
        embedding = np.random.rand(384).astype(np.float32)
        db.add_embedding(embedding)
        db.add_needs_embedding(embedding)
    
    # Mark indices dirty and rebuild
    db.mark_indices_dirty()
    db.ensure_indices_current()
    
    # Check indices
    main_index = db.get_main_index()
    needs_index = db.get_needs_index()
    
    assert main_index.ntotal == 3
    assert needs_index.ntotal == 3
    
    print("   ✓ Index management works")

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    db = DatabaseManager()
    
    # Test invalid company data
    try:
        db.add_company("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test missing required fields
    try:
        db.add_company({"name": "Incomplete"})
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    # Test invalid index operations
    try:
        db.get_company(-1)
        assert False, "Should return None for invalid index"
    except:
        pass
    
    try:
        db.update_company(999, {"name": "Test"})
        assert False, "Should have raised CompanyNotFoundError"
    except CompanyNotFoundError:
        pass
    
    try:
        db.delete_company(999)
        assert False, "Should have raised CompanyNotFoundError"
    except CompanyNotFoundError:
        pass
    
    print("   ✓ Error handling works")

def test_utility_methods():
    """Test utility methods"""
    print("Testing utility methods...")
    
    db = DatabaseManager()
    
    # Add test data
    for i in range(5):
        company_data = {
            "name": f"Company{i}",
            "industry": "Tech",
            "location": "USA",
            "revenue": 1000000,
            "team_size": 50,
            "founded": 2020,
            "website": f"https://company{i}.com",
            "description": f"Company {i}",
            "needs": f"Needs {i}",
            "challenges": f"Challenges {i}"
        }
        db.add_company(company_data)
    
    # Test get_all_companies
    all_companies = db.get_all_companies()
    assert len(all_companies) == 5
    assert all_companies[0]["name"] == "Company0"
    
    # Test get_all_indices
    indices = db.get_all_indices()
    assert indices == [0, 1, 2, 3, 4]
    
    # Test clear_all_data
    db.clear_all_data()
    assert db.get_company_count() == 0
    assert db.get_vector_count() == 0
    assert db.get_needs_vector_count() == 0
    
    print("   ✓ Utility methods work")

if __name__ == "__main__":
    print("=== DATABASE MANAGER TESTS ===")
    
    try:
        test_database_manager_initialization()
        test_company_crud_operations()
        test_vector_operations()
        test_index_management()
        test_error_handling()
        test_utility_methods()
        
        print("\n✅ ALL DATABASE MANAGER TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ DATABASE MANAGER TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)