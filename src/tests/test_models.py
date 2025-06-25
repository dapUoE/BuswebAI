#!/usr/bin/env python3
"""Test data models and exception classes"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DatabaseError, ValidationError, CompanyNotFoundError, EmbeddingError, Company

def test_exception_hierarchy():
    """Test exception class hierarchy"""
    print("Testing exception hierarchy...")
    
    # Test inheritance
    assert issubclass(ValidationError, DatabaseError)
    assert issubclass(CompanyNotFoundError, DatabaseError)
    assert issubclass(EmbeddingError, DatabaseError)
    assert issubclass(DatabaseError, Exception)
    print("   ✓ Exception hierarchy is correct")

def test_exception_creation():
    """Test exception creation and messages"""
    print("Testing exception creation...")
    
    # Test DatabaseError
    db_error = DatabaseError("Database failed")
    assert str(db_error) == "Database failed"
    assert isinstance(db_error, Exception)
    print("   ✓ DatabaseError works correctly")
    
    # Test ValidationError
    val_error = ValidationError("Invalid input")
    assert str(val_error) == "Invalid input"
    assert isinstance(val_error, DatabaseError)
    print("   ✓ ValidationError works correctly")
    
    # Test CompanyNotFoundError
    not_found_error = CompanyNotFoundError("Company not found")
    assert str(not_found_error) == "Company not found"
    assert isinstance(not_found_error, DatabaseError)
    print("   ✓ CompanyNotFoundError works correctly")
    
    # Test EmbeddingError
    embed_error = EmbeddingError("Embedding failed")
    assert str(embed_error) == "Embedding failed"
    assert isinstance(embed_error, DatabaseError)
    print("   ✓ EmbeddingError works correctly")

def test_company_dataclass():
    """Test Company dataclass functionality"""
    print("Testing Company dataclass...")
    
    # Test Company creation
    company = Company(
        name="TestCorp",
        industry="Technology",
        location="USA",
        revenue=1000000,
        team_size=50,
        founded=2020,
        website="https://testcorp.com",
        description="Test company",
        needs="Funding",
        challenges="Scaling"
    )
    
    # Test field access
    assert company.name == "TestCorp"
    assert company.industry == "Technology"
    assert company.location == "USA"
    assert company.revenue == 1000000
    assert company.team_size == 50
    assert company.founded == 2020
    assert company.website == "https://testcorp.com"
    assert company.description == "Test company"
    assert company.needs == "Funding"
    assert company.challenges == "Scaling"
    print("   ✓ Company field access works")

def test_company_to_dict():
    """Test Company to_dict method"""
    print("Testing Company to_dict...")
    
    company = Company(
        name="TestCorp",
        industry="Tech",
        location="USA", 
        revenue=1000000,
        team_size=50,
        founded=2020,
        website="https://test.com",
        description="Test",
        needs="Funding",
        challenges="Growth"
    )
    
    company_dict = company.to_dict()
    
    # Check it's a dictionary
    assert isinstance(company_dict, dict)
    
    # Check all fields are present
    expected_fields = [
        'name', 'industry', 'location', 'revenue', 'team_size',
        'founded', 'website', 'description', 'needs', 'challenges'
    ]
    
    for field in expected_fields:
        assert field in company_dict
        assert company_dict[field] == getattr(company, field)
    
    print("   ✓ to_dict method works correctly")

def test_company_from_dict():
    """Test Company from_dict class method"""
    print("Testing Company from_dict...")
    
    company_data = {
        'name': 'TestCorp',
        'industry': 'Tech',
        'location': 'USA',
        'revenue': 1000000,
        'team_size': 50,
        'founded': 2020,
        'website': 'https://test.com',
        'description': 'Test company',
        'needs': 'Funding',
        'challenges': 'Growth'
    }
    
    company = Company.from_dict(company_data)
    
    # Check it's a Company instance
    assert isinstance(company, Company)
    
    # Check all fields match
    for field, value in company_data.items():
        assert getattr(company, field) == value
    
    print("   ✓ from_dict method works correctly")

def test_company_roundtrip():
    """Test Company dict conversion roundtrip"""
    print("Testing Company dict roundtrip...")
    
    original = Company(
        name="RoundtripCorp",
        industry="Testing",
        location="Nowhere",
        revenue=500000,
        team_size=25,
        founded=2021,
        website="https://roundtrip.com",
        description="Testing roundtrip conversion",
        needs="Nothing",
        challenges="Everything"
    )
    
    # Convert to dict and back
    as_dict = original.to_dict()
    recreated = Company.from_dict(as_dict)
    
    # Should be equal
    assert recreated.name == original.name
    assert recreated.industry == original.industry
    assert recreated.location == original.location
    assert recreated.revenue == original.revenue
    assert recreated.team_size == original.team_size
    assert recreated.founded == original.founded
    assert recreated.website == original.website
    assert recreated.description == original.description
    assert recreated.needs == original.needs
    assert recreated.challenges == original.challenges
    
    print("   ✓ Dict conversion roundtrip works")

if __name__ == "__main__":
    print("=== MODELS TESTS ===")
    
    try:
        test_exception_hierarchy()
        test_exception_creation()
        test_company_dataclass()
        test_company_to_dict()
        test_company_from_dict()
        test_company_roundtrip()
        
        print("\n✅ ALL MODELS TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ MODELS TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)