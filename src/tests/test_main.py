#!/usr/bin/env python3
"""Test main interface and integration functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    add_company, get_company, get_all_companies, get_company_count,
    update_company, delete_company, create_company_profile
)
from search import company_db, vector_db, needs_vector_db
from models import ValidationError, DatabaseError, CompanyNotFoundError
from config import DatabaseConfig

def test_create_company_profile():
    """Test complete company profile creation"""
    print("Testing company profile creation...")
    
    # Clear existing data
    company_db.clear()
    vector_db.clear()
    needs_vector_db.clear()
    
    # Create a company profile
    company_idx = create_company_profile(
        name="IntegrationCorp",
        industry="Technology",
        location="USA",
        revenue=2000000,
        team_size=100,
        founded=2019,
        website="https://integrationcorp.com",
        description="Full-stack integration testing company",
        needs="Quality assurance partnerships",
        challenges="Maintaining test coverage across platforms"
    )
    
    assert isinstance(company_idx, int)
    assert company_idx == 0  # First company
    print(f"   ✓ Created company profile at index {company_idx}")
    
    # Verify company was added to all databases
    assert len(company_db) == 1
    assert len(vector_db) == 1
    assert len(needs_vector_db) == 1
    print("   ✓ Company added to all databases")
    
    # Verify company data
    company = get_company(company_idx)
    assert company["name"] == "IntegrationCorp"
    assert company["industry"] == "Technology"
    assert company["revenue"] == 2000000
    print("   ✓ Company data is correct")

def test_add_company():
    """Test adding company to database"""
    print("Testing company addition...")
    
    # Clear existing data
    company_db.clear()
    
    # Test valid company data
    company_data = {
        "name": "TestCorp",
        "industry": "Testing",
        "location": "TestLand",
        "revenue": 1000000,
        "team_size": 50,
        "founded": 2020,
        "website": "https://testcorp.com",
        "description": "A company for testing",
        "needs": "Better tests",
        "challenges": "Writing good tests"
    }
    
    company_idx = add_company(company_data)
    assert company_idx == 0
    assert len(company_db) == 1
    print("   ✓ Valid company added successfully")
    
    # Test missing required field
    invalid_data = {
        "name": "IncompleteCorp",
        "industry": "Incomplete"
        # Missing other required fields
    }
    
    try:
        add_company(invalid_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Missing required field" in str(e)
    
    print("   ✓ Missing field validation works")
    
    # Test invalid data type
    try:
        add_company("not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Company data must be a dictionary" in str(e)
    
    print("   ✓ Data type validation works")

def test_get_company():
    """Test company retrieval"""
    print("Testing company retrieval...")
    
    # Set up test data
    company_db.clear()
    test_data = {
        "name": "RetrieveCorp",
        "industry": "Retrieval",
        "location": "Database",
        "revenue": 500000,
        "team_size": 25,
        "founded": 2021,
        "website": "https://retrieve.com",
        "description": "Company for retrieval testing",
        "needs": "Better indexing",
        "challenges": "Fast queries"
    }
    
    company_idx = add_company(test_data)
    
    # Test valid retrieval
    retrieved = get_company(company_idx)
    assert retrieved["name"] == "RetrieveCorp"
    assert retrieved["revenue"] == 500000
    print("   ✓ Company retrieval works")

def test_get_all_companies():
    """Test getting all companies"""
    print("Testing get all companies...")
    
    # Clear and add test data
    company_db.clear()
    
    companies = [
        {"name": "Corp1", "industry": "Tech", "location": "USA", "revenue": 1000000, 
         "team_size": 50, "founded": 2020, "website": "https://corp1.com",
         "description": "First corp", "needs": "Growth", "challenges": "Scale"},
        {"name": "Corp2", "industry": "Health", "location": "UK", "revenue": 2000000,
         "team_size": 75, "founded": 2019, "website": "https://corp2.com",
         "description": "Second corp", "needs": "Compliance", "challenges": "Regulation"}
    ]
    
    for company in companies:
        add_company(company)
    
    all_companies = get_all_companies()
    assert len(all_companies) == 2
    assert all_companies[0]["name"] == "Corp1"
    assert all_companies[1]["name"] == "Corp2"
    print("   ✓ Get all companies works")

def test_get_company_count():
    """Test company count"""
    print("Testing company count...")
    
    # Clear data
    company_db.clear()
    assert get_company_count() == 0
    print("   ✓ Empty database count is 0")
    
    # Add companies
    for i in range(3):
        company_data = {
            "name": f"Corp{i}",
            "industry": "Tech",
            "location": "USA",
            "revenue": 1000000,
            "team_size": 50,
            "founded": 2020,
            "website": f"https://corp{i}.com",
            "description": f"Company {i}",
            "needs": "Growth",
            "challenges": "Scale"
        }
        add_company(company_data)
    
    assert get_company_count() == 3
    print("   ✓ Company count after additions is correct")

def test_update_company():
    """Test company updating"""
    print("Testing company updating...")
    
    # Set up initial company
    company_db.clear()
    vector_db.clear()
    needs_vector_db.clear()
    
    original_data = {
        "name": "UpdateCorp",
        "industry": "Original",
        "location": "OldPlace",
        "revenue": 1000000,
        "team_size": 50,
        "founded": 2020,
        "website": "https://old.com",
        "description": "Original description",
        "needs": "Original needs",
        "challenges": "Original challenges"
    }
    
    company_idx = add_company(original_data)
    
    # Update company
    updated_data = {
        "name": "UpdatedCorp",
        "industry": "Updated",
        "location": "NewPlace",
        "revenue": 2000000,
        "team_size": 100,
        "founded": 2020,  # Keep same
        "website": "https://new.com",
        "description": "Updated description with new features",
        "needs": "Updated needs for growth",
        "challenges": "Updated challenges and solutions"
    }
    
    result = update_company(company_idx, updated_data)
    assert result == True
    print("   ✓ Company update succeeded")
    
    # Verify update
    updated_company = get_company(company_idx)
    assert updated_company["name"] == "UpdatedCorp"
    assert updated_company["revenue"] == 2000000
    assert updated_company["description"] == "Updated description with new features"
    print("   ✓ Updated data is correct")
    
    # Test updating non-existent company
    try:
        update_company(999, updated_data)
        assert False, "Should have raised CompanyNotFoundError"
    except CompanyNotFoundError as e:
        assert "not found" in str(e)
    
    print("   ✓ Non-existent company update validation works")
    
    # Test invalid data
    try:
        update_company(company_idx, "not a dict")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Company data must be a dictionary" in str(e)
    
    try:
        incomplete_data = {"name": "Incomplete"}  # Missing required fields
        update_company(company_idx, incomplete_data)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Missing required field" in str(e)
    
    print("   ✓ Update validation works")

def test_delete_company():
    """Test company deletion"""
    print("Testing company deletion...")
    
    # Set up test companies
    company_db.clear()
    vector_db.clear()
    needs_vector_db.clear()
    
    companies = []
    for i in range(3):
        company_data = {
            "name": f"DeleteCorp{i}",
            "industry": "Delete",
            "location": "Trash",
            "revenue": 1000000,
            "team_size": 50,
            "founded": 2020,
            "website": f"https://delete{i}.com",
            "description": f"Company {i} for deletion",
            "needs": "To be deleted",
            "challenges": "Deletion testing"
        }
        idx = add_company(company_data)
        companies.append(idx)
    
    assert len(company_db) == 3
    assert get_company_count() == 3
    print("   ✓ Test companies set up")
    
    # Delete middle company
    result = delete_company(1)
    assert result == True
    assert len(company_db) == 2
    assert get_company_count() == 2
    print("   ✓ Company deletion succeeded")
    
    # Verify remaining companies shifted indices
    remaining = get_all_companies()
    assert len(remaining) == 2
    assert remaining[0]["name"] == "DeleteCorp0"
    assert remaining[1]["name"] == "DeleteCorp2"
    print("   ✓ Remaining companies correct after deletion")
    
    # Test deleting non-existent company
    try:
        delete_company(999)
        assert False, "Should have raised CompanyNotFoundError"
    except CompanyNotFoundError as e:
        assert "not found" in str(e)
    
    print("   ✓ Non-existent company deletion validation works")
    
    # Test invalid index type
    try:
        delete_company("not an int")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "Index must be an integer" in str(e)
    
    print("   ✓ Delete validation works")

def test_integration_workflow():
    """Test complete CRUD workflow"""
    print("Testing complete CRUD workflow...")
    
    # Clear all data
    company_db.clear()
    vector_db.clear()
    needs_vector_db.clear()
    
    # CREATE - Add multiple companies
    companies_data = [
        {
            "name": "WorkflowCorp1",
            "industry": "Workflow",
            "location": "ProcessLand",
            "revenue": 1000000,
            "team_size": 50,
            "founded": 2020,
            "website": "https://workflow1.com",
            "description": "First workflow company",
            "needs": "Process optimization",
            "challenges": "Workflow efficiency"
        },
        {
            "name": "WorkflowCorp2",
            "industry": "Workflow",
            "location": "ProcessLand",
            "revenue": 2000000,
            "team_size": 75,
            "founded": 2019,
            "website": "https://workflow2.com",
            "description": "Second workflow company",
            "needs": "Automation tools",
            "challenges": "Legacy system integration"
        }
    ]
    
    created_indices = []
    for company_data in companies_data:
        idx = add_company(company_data)
        created_indices.append(idx)
    
    assert len(created_indices) == 2
    assert get_company_count() == 2
    print("   ✓ CREATE: Companies added successfully")
    
    # READ - Retrieve companies
    all_companies = get_all_companies()
    assert len(all_companies) == 2
    
    first_company = get_company(created_indices[0])
    assert first_company["name"] == "WorkflowCorp1"
    print("   ✓ READ: Companies retrieved successfully")
    
    # UPDATE - Modify first company
    updated_data = first_company.copy()
    updated_data["revenue"] = 1500000
    updated_data["description"] = "Updated first workflow company"
    
    update_result = update_company(created_indices[0], updated_data)
    assert update_result == True
    
    updated_company = get_company(created_indices[0])
    assert updated_company["revenue"] == 1500000
    assert updated_company["description"] == "Updated first workflow company"
    print("   ✓ UPDATE: Company modified successfully")
    
    # DELETE - Remove second company
    delete_result = delete_company(created_indices[1])
    assert delete_result == True
    assert get_company_count() == 1
    
    remaining_companies = get_all_companies()
    assert len(remaining_companies) == 1
    assert remaining_companies[0]["name"] == "WorkflowCorp1"
    print("   ✓ DELETE: Company removed successfully")
    
    print("   ✅ Complete CRUD workflow successful")

def test_validation_edge_cases():
    """Test validation edge cases"""
    print("Testing validation edge cases...")
    
    company_db.clear()
    
    # Test boundary values
    boundary_company = {
        "name": "BoundaryCorp",
        "industry": "Testing",
        "location": "EdgeCase",
        "revenue": DatabaseConfig.MIN_REVENUE,  # Minimum allowed
        "team_size": DatabaseConfig.MIN_TEAM_SIZE,  # Minimum allowed
        "founded": DatabaseConfig.MIN_FOUNDED_YEAR,  # Minimum allowed
        "website": "https://boundary.com",
        "description": "Testing boundary values",
        "needs": "Edge case testing",
        "challenges": "Boundary validation"
    }
    
    try:
        idx = add_company(boundary_company)
        print("   ✓ Boundary values accepted")
    except Exception as e:
        assert False, f"Boundary values should be valid: {e}"
    
    # Test maximum values
    max_company = {
        "name": "MaxCorp",
        "industry": "Testing",
        "location": "MaxLand",
        "revenue": DatabaseConfig.MAX_REVENUE,
        "team_size": DatabaseConfig.MAX_TEAM_SIZE,
        "founded": DatabaseConfig.MAX_FOUNDED_YEAR,
        "website": "https://max.com",
        "description": "Testing maximum values",
        "needs": "Maximum testing",
        "challenges": "Maximum validation"
    }
    
    try:
        idx = add_company(max_company)
        print("   ✓ Maximum values accepted")
    except Exception as e:
        assert False, f"Maximum values should be valid: {e}"

if __name__ == "__main__":
    print("=== MAIN INTERFACE TESTS ===")
    
    try:
        test_create_company_profile()
        test_add_company()
        test_get_company()
        test_get_all_companies()
        test_get_company_count()
        test_update_company()
        test_delete_company()
        test_integration_workflow()
        test_validation_edge_cases()
        
        print("\n✅ ALL MAIN INTERFACE TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ MAIN INTERFACE TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)