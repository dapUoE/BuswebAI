#!/usr/bin/env python3
"""Test input validation functions"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validators import validate_string_field, validate_integer_field, validate_company_data
from models import ValidationError
from config import DatabaseConfig

def test_validate_string_field():
    """Test string field validation"""
    print("Testing string field validation...")
    
    # Valid strings
    assert validate_string_field("Test", "name") == "Test"
    assert validate_string_field("  Test  ", "name") == "Test"  # Trimmed
    print("   ✓ Valid strings accepted")
    
    # Empty string with required=False
    assert validate_string_field("", "optional", required=False) == ""
    assert validate_string_field(None, "optional", required=False) == ""
    print("   ✓ Optional empty strings handled")
    
    # Invalid inputs should raise ValidationError
    try:
        validate_string_field(None, "name", required=True)
        assert False, "Should have raised ValidationError for None"
    except ValidationError as e:
        assert "name is required" in str(e)
    
    try:
        validate_string_field(123, "name")
        assert False, "Should have raised ValidationError for non-string"
    except ValidationError as e:
        assert "name must be a string" in str(e)
    
    try:
        validate_string_field("", "name", required=True)
        assert False, "Should have raised ValidationError for empty required string"
    except ValidationError as e:
        assert "name cannot be empty" in str(e)
    
    try:
        long_string = "x" * (DatabaseConfig.MAX_STRING_LENGTH + 1)
        validate_string_field(long_string, "name")
        assert False, "Should have raised ValidationError for too long string"
    except ValidationError as e:
        assert "too long" in str(e)
    
    print("   ✓ Invalid strings properly rejected")

def test_validate_integer_field():
    """Test integer field validation"""
    print("Testing integer field validation...")
    
    # Valid integers
    assert validate_integer_field(42, "test", 0, 100) == 42
    assert validate_integer_field(42.0, "test", 0, 100) == 42  # Float converted to int
    assert validate_integer_field(0, "test", 0, 100) == 0  # Min boundary
    assert validate_integer_field(100, "test", 0, 100) == 100  # Max boundary
    print("   ✓ Valid integers accepted")
    
    # Invalid inputs should raise ValidationError
    try:
        validate_integer_field(None, "test", 0, 100)
        assert False, "Should have raised ValidationError for None"
    except ValidationError as e:
        assert "test is required" in str(e)
    
    try:
        validate_integer_field("not a number", "test", 0, 100)
        assert False, "Should have raised ValidationError for string"
    except ValidationError as e:
        assert "test must be a number" in str(e)
    
    try:
        validate_integer_field(-1, "test", 0, 100)
        assert False, "Should have raised ValidationError for below min"
    except ValidationError as e:
        assert "must be between 0 and 100" in str(e)
    
    try:
        validate_integer_field(101, "test", 0, 100)
        assert False, "Should have raised ValidationError for above max"
    except ValidationError as e:
        assert "must be between 0 and 100" in str(e)
    
    print("   ✓ Invalid integers properly rejected")

def test_validate_company_data():
    """Test complete company data validation"""
    print("Testing company data validation...")
    
    # Valid company data
    valid_data = validate_company_data(
        name="TestCorp",
        industry="Technology",
        location="USA",
        revenue=1000000,
        team_size=50,
        founded=2020,
        website="https://testcorp.com",
        description="A test company",
        needs="Funding and partnerships",
        challenges="Scaling and growth"
    )
    
    # Check all fields are present and validated
    expected_fields = [
        'name', 'industry', 'location', 'revenue', 'team_size',
        'founded', 'website', 'description', 'needs', 'challenges'
    ]
    
    for field in expected_fields:
        assert field in valid_data
    
    # Check string fields are trimmed
    assert valid_data['name'] == "TestCorp"
    assert valid_data['industry'] == "Technology"
    
    # Check integer fields are converted
    assert isinstance(valid_data['revenue'], int)
    assert isinstance(valid_data['team_size'], int)
    assert isinstance(valid_data['founded'], int)
    
    print("   ✓ Valid company data accepted")

def test_validate_company_data_errors():
    """Test company data validation error cases"""
    print("Testing company data validation errors...")
    
    # Missing name
    try:
        validate_company_data(
            name=None,
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
        assert False, "Should have raised ValidationError for missing name"
    except ValidationError as e:
        assert "name is required" in str(e)
    
    # Invalid revenue (negative)
    try:
        validate_company_data(
            name="TestCorp",
            industry="Tech",
            location="USA",
            revenue=-1000,
            team_size=50,
            founded=2020,
            website="https://test.com",
            description="Test",
            needs="Funding",
            challenges="Growth"
        )
        assert False, "Should have raised ValidationError for negative revenue"
    except ValidationError as e:
        assert "revenue must be between" in str(e)
    
    # Invalid team size (zero)
    try:
        validate_company_data(
            name="TestCorp",
            industry="Tech",
            location="USA",
            revenue=1000000,
            team_size=0,
            founded=2020,
            website="https://test.com",
            description="Test",
            needs="Funding",
            challenges="Growth"
        )
        assert False, "Should have raised ValidationError for zero team size"
    except ValidationError as e:
        assert "team_size must be between" in str(e)
    
    # Invalid founded year (future)
    try:
        validate_company_data(
            name="TestCorp",
            industry="Tech",
            location="USA",
            revenue=1000000,
            team_size=50,
            founded=2150,
            website="https://test.com",
            description="Test",
            needs="Funding",
            challenges="Growth"
        )
        assert False, "Should have raised ValidationError for future founded year"
    except ValidationError as e:
        assert "founded must be between" in str(e)
    
    print("   ✓ Invalid company data properly rejected")

def test_validation_ranges():
    """Test validation with config range constants"""
    print("Testing validation ranges...")
    
    # Test revenue ranges
    min_revenue_company = validate_company_data(
        "MinRevCorp", "Tech", "USA", DatabaseConfig.MIN_REVENUE, 10, 2020,
        "https://test.com", "Test", "Funding", "Growth"
    )
    assert min_revenue_company['revenue'] == DatabaseConfig.MIN_REVENUE
    
    # Test team size ranges
    min_team_company = validate_company_data(
        "MinTeamCorp", "Tech", "USA", 100000, DatabaseConfig.MIN_TEAM_SIZE, 2020,
        "https://test.com", "Test", "Funding", "Growth"
    )
    assert min_team_company['team_size'] == DatabaseConfig.MIN_TEAM_SIZE
    
    # Test founded year ranges
    old_company = validate_company_data(
        "OldCorp", "Tech", "USA", 100000, 10, DatabaseConfig.MIN_FOUNDED_YEAR,
        "https://test.com", "Test", "Funding", "Growth"
    )
    assert old_company['founded'] == DatabaseConfig.MIN_FOUNDED_YEAR
    
    print("   ✓ Validation ranges work with config constants")

if __name__ == "__main__":
    print("=== VALIDATORS TESTS ===")
    
    try:
        test_validate_string_field()
        test_validate_integer_field()
        test_validate_company_data()
        test_validate_company_data_errors()
        test_validation_ranges()
        
        print("\n✅ ALL VALIDATORS TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ VALIDATORS TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)