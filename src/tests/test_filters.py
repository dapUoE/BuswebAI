#!/usr/bin/env python3
"""Test filtering functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filters import (
    filter_by_revenue_range, filter_by_team_size_range, filter_by_founded_range,
    filter_by_industry, filter_by_location, filter_by_name_contains,
    filter_by_website_domain, get_all_company_indices, apply_filters,
    search_companies_with_filters, filter_companies
)
from search import company_db
from database_core import append_to_db
from models import ValidationError, DatabaseError

def setup_test_companies():
    """Set up test companies for filtering"""
    # Clear existing data
    company_db.clear()
    
    # Create diverse test companies
    companies = [
        {
            "name": "BigTech Corp",
            "industry": "Technology",
            "location": "USA",
            "revenue": 10000000,  # 10M
            "team_size": 500,
            "founded": 2010,
            "website": "https://bigtech.com",
            "description": "Large technology corporation",
            "needs": "Global expansion",
            "challenges": "Market competition"
        },
        {
            "name": "StartupAI",
            "industry": "AI/ML",
            "location": "USA",
            "revenue": 500000,  # 500K
            "team_size": 15,
            "founded": 2020,
            "website": "https://startupai.io",
            "description": "AI startup for small businesses",
            "needs": "Seed funding",
            "challenges": "Product-market fit"
        },
        {
            "name": "HealthTech GmbH",
            "industry": "Healthcare",
            "location": "Germany",
            "revenue": 2000000,  # 2M
            "team_size": 80,
            "founded": 2015,
            "website": "https://healthtech.de",
            "description": "Medical device software",
            "needs": "Regulatory approval",
            "challenges": "Compliance requirements"
        },
        {
            "name": "FinanceBot Ltd",
            "industry": "FinTech",
            "location": "UK",
            "revenue": 1500000,  # 1.5M
            "team_size": 45,
            "founded": 2018,
            "website": "https://financebot.co.uk",
            "description": "Automated trading platform",
            "needs": "Banking partnerships",
            "challenges": "Financial regulations"
        },
        {
            "name": "MedAI Solutions",
            "industry": "AI/ML",
            "location": "Germany",
            "revenue": 800000,  # 800K
            "team_size": 25,
            "founded": 2019,
            "website": "https://medai.de",
            "description": "AI for medical diagnosis",
            "needs": "Hospital partnerships",
            "challenges": "Data privacy laws"
        }
    ]
    
    # Add companies to database
    for company in companies:
        append_to_db(company, company_db)
    
    return companies

def test_filter_by_revenue_range():
    """Test revenue range filtering"""
    print("Testing revenue range filtering...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    # Filter for companies with revenue > 1M
    high_revenue = filter_by_revenue_range(all_indices, min_revenue=1000000)
    assert len(high_revenue) == 3  # BigTech (10M), HealthTech (2M), FinanceBot (1.5M)
    print("   ✓ Min revenue filter works")
    
    # Filter for companies with revenue < 1M
    low_revenue = filter_by_revenue_range(all_indices, max_revenue=999999)
    assert len(low_revenue) == 2  # StartupAI (500K), MedAI (800K)
    print("   ✓ Max revenue filter works")
    
    # Filter for companies with revenue between 1M and 5M
    mid_revenue = filter_by_revenue_range(all_indices, min_revenue=1000000, max_revenue=5000000)
    assert len(mid_revenue) == 2  # HealthTech (2M), FinanceBot (1.5M)
    print("   ✓ Revenue range filter works")
    
    # No filters should return all
    no_filter = filter_by_revenue_range(all_indices)
    assert len(no_filter) == 5
    print("   ✓ No revenue filter returns all companies")

def test_filter_by_team_size_range():
    """Test team size range filtering"""
    print("Testing team size range filtering...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    # Small teams (< 50)
    small_teams = filter_by_team_size_range(all_indices, max_size=49)
    assert len(small_teams) == 3  # StartupAI (15), FinanceBot (45), MedAI (25)
    print("   ✓ Small team filter works")
    
    # Large teams (> 50)
    large_teams = filter_by_team_size_range(all_indices, min_size=50)
    assert len(large_teams) == 2  # BigTech (500), HealthTech (80)
    print("   ✓ Large team filter works")
    
    # Medium teams (20-100)
    medium_teams = filter_by_team_size_range(all_indices, min_size=20, max_size=100)
    assert len(medium_teams) == 3  # FinanceBot (45), HealthTech (80), MedAI (25)
    print("   ✓ Team size range filter works")

def test_filter_by_founded_range():
    """Test founded year range filtering"""
    print("Testing founded year range filtering...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    # Old companies (founded before 2018)
    old_companies = filter_by_founded_range(all_indices, max_year=2017)
    assert len(old_companies) == 2  # BigTech (2010), HealthTech (2015)
    print("   ✓ Old companies filter works")
    
    # New companies (founded after 2018)
    new_companies = filter_by_founded_range(all_indices, min_year=2019)
    assert len(new_companies) == 2  # StartupAI (2020), MedAI (2019)
    print("   ✓ New companies filter works")
    
    # Companies founded in 2018-2020
    recent_companies = filter_by_founded_range(all_indices, min_year=2018, max_year=2020)
    assert len(recent_companies) == 3  # FinanceBot (2018), MedAI (2019), StartupAI (2020)
    print("   ✓ Founded year range filter works")

def test_filter_by_industry():
    """Test industry filtering"""
    print("Testing industry filtering...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    # Single industry
    ai_companies = filter_by_industry(all_indices, "AI/ML")
    assert len(ai_companies) == 2  # StartupAI, MedAI
    print("   ✓ Single industry filter works")
    
    # Multiple industries
    tech_companies = filter_by_industry(all_indices, ["Technology", "AI/ML"])
    assert len(tech_companies) == 3  # BigTech, StartupAI, MedAI
    print("   ✓ Multiple industry filter works")
    
    # Case insensitive
    healthcare = filter_by_industry(all_indices, "healthcare")  # lowercase
    assert len(healthcare) == 1  # HealthTech
    print("   ✓ Case insensitive industry filter works")

def test_filter_by_location():
    """Test location filtering"""
    print("Testing location filtering...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    # Single location
    usa_companies = filter_by_location(all_indices, "USA")
    assert len(usa_companies) == 2  # BigTech, StartupAI
    print("   ✓ Single location filter works")
    
    # Multiple locations
    europe_companies = filter_by_location(all_indices, ["Germany", "UK"])
    assert len(europe_companies) == 3  # HealthTech, FinanceBot, MedAI
    print("   ✓ Multiple location filter works")
    
    # Case insensitive
    germany_companies = filter_by_location(all_indices, "germany")  # lowercase
    assert len(germany_companies) == 2  # HealthTech, MedAI
    print("   ✓ Case insensitive location filter works")

def test_filter_by_name_contains():
    """Test name substring filtering"""
    print("Testing name substring filtering...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    # Contains "Tech"
    tech_names = filter_by_name_contains(all_indices, "Tech")
    assert len(tech_names) == 2  # BigTech Corp, HealthTech GmbH
    print("   ✓ Name contains filter works")
    
    # Contains "AI" (case insensitive)
    ai_names = filter_by_name_contains(all_indices, "ai")
    assert len(ai_names) == 2  # StartupAI, MedAI Solutions
    print("   ✓ Case insensitive name filter works")
    
    # No matches
    no_matches = filter_by_name_contains(all_indices, "NonExistent")
    assert len(no_matches) == 0
    print("   ✓ Name filter with no matches works")

def test_filter_by_website_domain():
    """Test website domain filtering"""
    print("Testing website domain filtering...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    # .com domains
    com_domains = filter_by_website_domain(all_indices, ".com")
    assert len(com_domains) == 1  # BigTech only (.io is not .com)
    print("   ✓ .com domain filter works")
    
    # .de domains
    de_domains = filter_by_website_domain(all_indices, ".de")
    assert len(de_domains) == 2  # HealthTech, MedAI
    print("   ✓ .de domain filter works")
    
    # .io domains
    io_domains = filter_by_website_domain(all_indices, ".io")
    assert len(io_domains) == 1  # StartupAI
    print("   ✓ .io domain filter works")

def test_get_all_company_indices():
    """Test getting all company indices"""
    print("Testing get all company indices...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    assert len(all_indices) == 5
    assert all_indices == [0, 1, 2, 3, 4]
    print("   ✓ All company indices returned correctly")

def test_apply_filters():
    """Test applying multiple filters"""
    print("Testing multiple filter application...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    # Filter for German AI/ML companies
    filters = {
        "location": "Germany",
        "industry": "AI/ML"
    }
    result = apply_filters(all_indices, filters)
    assert len(result) == 1  # MedAI Solutions
    print("   ✓ Location + Industry filter works")
    
    # Filter for small recent companies
    filters = {
        "max_team_size": 50,
        "min_founded": 2019
    }
    result = apply_filters(all_indices, filters)
    assert len(result) == 2  # StartupAI (15), MedAI (25)
    print("   ✓ Team size + Founded year filter works")
    
    # Complex filter: USA tech companies with high revenue
    filters = {
        "location": "USA",
        "industry": ["Technology", "AI/ML"],
        "min_revenue": 1000000
    }
    result = apply_filters(all_indices, filters)
    assert len(result) == 1  # BigTech Corp
    print("   ✓ Complex multi-filter works")
    
    # No filters should return all
    result = apply_filters(all_indices, {})
    assert len(result) == 5
    print("   ✓ No filters returns all companies")

def test_filter_companies():
    """Test high-level filter_companies function"""
    print("Testing filter_companies function...")
    
    companies = setup_test_companies()
    
    # Filter by industry
    tech_companies = filter_companies(industry="Technology")
    assert len(tech_companies) == 1
    assert tech_companies[0]["name"] == "BigTech Corp"
    print("   ✓ filter_companies by industry works")
    
    # Filter by multiple criteria
    german_companies = filter_companies(location="Germany", min_team_size=50)
    assert len(german_companies) == 1
    assert german_companies[0]["name"] == "HealthTech GmbH"
    print("   ✓ filter_companies with multiple criteria works")
    
    # Check result structure
    for company in german_companies:
        expected_fields = [
            "name", "match_score", "description", "needs", "challenges",
            "website", "industry", "location", "revenue", "team_size", "founded"
        ]
        for field in expected_fields:
            assert field in company
        assert company["match_score"] is None  # No semantic matching
    
    print("   ✓ filter_companies result structure is correct")

def test_search_companies_with_filters():
    """Test combined search and filtering"""
    print("Testing search with filters...")
    
    companies = setup_test_companies()
    
    # Pure filtering (no text query)
    filtered_only = search_companies_with_filters(
        text_query=None,
        industry="AI/ML",
        min_team_size=20
    )
    assert len(filtered_only) == 1  # MedAI (25 team members)
    assert filtered_only[0]["name"] == "MedAI Solutions"
    assert filtered_only[0]["match_score"] is None  # No semantic scoring
    print("   ✓ Pure filtering works")
    
    # Note: Text query + filters would require actual embeddings setup,
    # which is complex for this test. The search module tests cover that.
    
    print("   ✓ search_companies_with_filters works")

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("Testing edge cases...")
    
    companies = setup_test_companies()
    all_indices = get_all_company_indices()
    
    # Empty index list
    empty_result = filter_by_revenue_range([], min_revenue=1000000)
    assert empty_result == []
    print("   ✓ Empty index list returns empty result")
    
    # Filters that match nothing
    no_matches = filter_by_revenue_range(all_indices, min_revenue=100000000)  # 100M
    assert no_matches == []
    print("   ✓ Impossible filter returns empty result")
    
    # Invalid company index (should be handled gracefully)
    invalid_indices = [0, 1, 999]  # 999 doesn't exist
    filtered = filter_by_revenue_range(invalid_indices, min_revenue=0)
    assert len(filtered) == 2  # Only valid indices 0, 1
    print("   ✓ Invalid indices handled gracefully")

if __name__ == "__main__":
    print("=== FILTERS TESTS ===")
    
    try:
        test_filter_by_revenue_range()
        test_filter_by_team_size_range()
        test_filter_by_founded_range()
        test_filter_by_industry()
        test_filter_by_location()
        test_filter_by_name_contains()
        test_filter_by_website_domain()
        test_get_all_company_indices()
        test_apply_filters()
        test_filter_companies()
        test_search_companies_with_filters()
        test_edge_cases()
        
        print("\n✅ ALL FILTERS TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ FILTERS TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)