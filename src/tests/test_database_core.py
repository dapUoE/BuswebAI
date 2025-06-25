#!/usr/bin/env python3
"""Test core database operations"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_core import (
    append_to_db, get_db_length, get_last_index, is_index_valid, 
    is_index_in_range, is_valid_index, get_item_by_index, get_company,
    copy_db, get_field, get_name, get_industry, get_location, 
    get_revenue, get_team_size, get_founded, get_website, 
    get_description, get_needs, get_challenges, create_company_dict
)
from models import ValidationError, DatabaseError

def test_database_operations():
    """Test basic database operations"""
    print("Testing basic database operations...")
    
    # Start with empty database
    test_db = []
    
    # Test get_db_length
    assert get_db_length(test_db) == 0
    print("   ✓ Empty database length is 0")
    
    # Test append_to_db
    test_data = {"name": "TestCorp", "industry": "Tech"}
    append_to_db(test_data, test_db)
    assert get_db_length(test_db) == 1
    print("   ✓ Append to database works")
    
    # Test get_last_index
    assert get_last_index(test_db) == 0
    print("   ✓ Last index is correct")
    
    # Add more data
    append_to_db({"name": "Corp2", "industry": "Finance"}, test_db)
    assert get_db_length(test_db) == 2
    assert get_last_index(test_db) == 1
    print("   ✓ Multiple items handled correctly")

def test_index_validation():
    """Test index validation functions"""
    print("Testing index validation...")
    
    test_db = [{"id": 1}, {"id": 2}, {"id": 3}]
    
    # Test is_index_valid
    assert is_index_valid(0) == True
    assert is_index_valid(1) == True
    assert is_index_valid(-1) == False
    assert is_index_valid("not an int") == False
    print("   ✓ is_index_valid works correctly")
    
    # Test is_index_in_range
    assert is_index_in_range(0, test_db) == True
    assert is_index_in_range(2, test_db) == True
    assert is_index_in_range(3, test_db) == False
    assert is_index_in_range(-1, test_db) == False
    assert is_index_in_range("not an int", test_db) == False
    print("   ✓ is_index_in_range works correctly")
    
    # Test is_valid_index (combines both checks)
    assert is_valid_index(0, test_db) == True
    assert is_valid_index(2, test_db) == True
    assert is_valid_index(3, test_db) == False
    assert is_valid_index(-1, test_db) == False
    print("   ✓ is_valid_index works correctly")

def test_item_retrieval():
    """Test item retrieval functions"""
    print("Testing item retrieval...")
    
    test_db = [
        {"name": "Corp1", "value": 100},
        {"name": "Corp2", "value": 200},
        {"name": "Corp3", "value": 300}
    ]
    
    # Test get_item_by_index
    item0 = get_item_by_index(0, test_db)
    assert item0 == {"name": "Corp1", "value": 100}
    
    item2 = get_item_by_index(2, test_db)
    assert item2 == {"name": "Corp3", "value": 300}
    
    # Invalid index should return None
    invalid_item = get_item_by_index(5, test_db)
    assert invalid_item is None
    
    # Test error for non-integer index
    try:
        get_item_by_index("not an int", test_db)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
    
    print("   ✓ Item retrieval works correctly")
    
    # Test get_company (wrapper function)
    company = get_company(1, test_db)
    assert company == {"name": "Corp2", "value": 200}
    print("   ✓ get_company wrapper works")

def test_database_copy():
    """Test database copying"""
    print("Testing database copying...")
    
    original_db = [{"id": 1}, {"id": 2}]
    copied_db = copy_db(original_db)
    
    # Should be equal but different objects
    assert copied_db == original_db
    assert copied_db is not original_db
    
    # Modifying copy shouldn't affect original
    copied_db.append({"id": 3})
    assert len(original_db) == 2
    assert len(copied_db) == 3
    
    print("   ✓ Database copying works correctly")

def test_field_access():
    """Test generic field access"""
    print("Testing field access...")
    
    test_company = {
        "name": "TestCorp",
        "industry": "Technology",
        "location": "USA",
        "revenue": 1000000,
        "team_size": 50,
        "founded": 2020,
        "website": "https://testcorp.com",
        "description": "A test company",
        "needs": "Funding",
        "challenges": "Scaling"
    }
    
    # Test generic get_field
    assert get_field(test_company, "name") == "TestCorp"
    assert get_field(test_company, "revenue") == 1000000
    print("   ✓ Generic field access works")
    
    # Test field not found
    try:
        get_field(test_company, "nonexistent")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "missing nonexistent field" in str(e)
    
    # Test invalid company (not dict)
    try:
        get_field("not a dict", "name")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Company must be a dictionary" in str(e)
    
    print("   ✓ Field access validation works")

def test_convenience_field_accessors():
    """Test convenience field accessor functions"""
    print("Testing convenience field accessors...")
    
    test_company = {
        "name": "TestCorp",
        "industry": "Technology", 
        "location": "USA",
        "revenue": 1000000,
        "team_size": 50,
        "founded": 2020,
        "website": "https://testcorp.com",
        "description": "A test company",
        "needs": "Funding and partnerships",
        "challenges": "Scaling rapidly"
    }
    
    # Test all convenience accessors
    assert get_name(test_company) == "TestCorp"
    assert get_industry(test_company) == "Technology"
    assert get_location(test_company) == "USA"
    assert get_revenue(test_company) == 1000000
    assert get_team_size(test_company) == 50
    assert get_founded(test_company) == 2020
    assert get_website(test_company) == "https://testcorp.com"
    assert get_description(test_company) == "A test company"
    assert get_needs(test_company) == "Funding and partnerships"
    assert get_challenges(test_company) == "Scaling rapidly"
    
    print("   ✓ All convenience field accessors work")

def test_create_company_dict():
    """Test company dictionary creation"""
    print("Testing company dictionary creation...")
    
    company_dict = create_company_dict(
        name="NewCorp",
        industry="FinTech",
        location="UK",
        revenue=500000,
        team_size=25,
        founded=2021,
        website="https://newcorp.co.uk",
        description="A new fintech company",
        needs="Banking partnerships",
        challenges="Regulatory compliance"
    )
    
    # Should be a dictionary with all fields
    assert isinstance(company_dict, dict)
    assert company_dict["name"] == "NewCorp"
    assert company_dict["industry"] == "FinTech"
    assert company_dict["location"] == "UK"
    assert company_dict["revenue"] == 500000
    assert company_dict["team_size"] == 25
    assert company_dict["founded"] == 2021
    assert company_dict["website"] == "https://newcorp.co.uk"
    assert company_dict["description"] == "A new fintech company"
    assert company_dict["needs"] == "Banking partnerships"
    assert company_dict["challenges"] == "Regulatory compliance"
    
    print("   ✓ Company dictionary creation works")

def test_error_handling():
    """Test error handling in database operations"""
    print("Testing error handling...")
    
    # Test append_to_db with invalid data
    test_db = []
    try:
        append_to_db("not a dict", test_db)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Data must be a dictionary" in str(e)
    
    # Test get_last_index with empty database
    empty_db = []
    try:
        get_last_index(empty_db)
        assert False, "Should have raised DatabaseError"
    except DatabaseError as e:
        assert "Database is empty" in str(e)
    
    print("   ✓ Error handling works correctly")

if __name__ == "__main__":
    print("=== DATABASE CORE TESTS ===")
    
    try:
        test_database_operations()
        test_index_validation()
        test_item_retrieval()
        test_database_copy()
        test_field_access()
        test_convenience_field_accessors()
        test_create_company_dict()
        test_error_handling()
        
        print("\n✅ ALL DATABASE CORE TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ DATABASE CORE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)