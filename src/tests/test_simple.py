#!/usr/bin/env python3
"""Simple test runner for the modular database system"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main interface
from main import create_company_profile, search_companies_by_text, search_companies_by_needs, filter_companies, search_companies_with_filters

def run_basic_tests():
    """Run basic functionality tests"""
    print("=== BASIC FUNCTIONALITY TESTS ===")
    
    try:
        # Test 1: Create company profiles
        print("1. Testing company creation...")
        idx1 = create_company_profile(
            name="TestCorp",
            industry="Technology",
            location="USA",
            revenue=1000000,
            team_size=50,
            founded=2020,
            website="https://testcorp.com",
            description="A test technology company",
            needs="Looking for partnerships",
            challenges="Scaling rapidly"
        )
        print(f"   ✓ Created company at index {idx1}")
        
        idx2 = create_company_profile(
            name="HealthTech Inc",
            industry="Healthcare",
            location="Germany",
            revenue=500000,
            team_size=25,
            founded=2021,
            website="https://healthtech.de",
            description="Healthcare technology solutions",
            needs="Seeking regulatory approval",
            challenges="Complex compliance requirements"
        )
        print(f"   ✓ Created company at index {idx2}")
        
        # Test 2: Text search
        print("\n2. Testing semantic search...")
        results = search_companies_by_text("technology partnerships")
        print(f"   ✓ Found {len(results)} results for 'technology partnerships'")
        for r in results:
            print(f"     - {r['name']}: {r['match_score']}")
        
        # Test 3: Needs search
        print("\n3. Testing needs-specific search...")
        needs_results = search_companies_by_needs("partnerships approval")
        print(f"   ✓ Found {len(needs_results)} results for 'partnerships approval'")
        for r in needs_results:
            print(f"     - {r['name']}: {r['match_score']}")
        
        # Test 4: Filtering
        print("\n4. Testing filtering...")
        tech_companies = filter_companies(industry="Technology")
        print(f"   ✓ Found {len(tech_companies)} Technology companies")
        
        high_revenue = filter_companies(min_revenue=750000)
        print(f"   ✓ Found {len(high_revenue)} companies with >$750k revenue")
        
        # Test 5: Combined search + filters
        print("\n5. Testing combined search + filters...")
        combined = search_companies_with_filters(
            text_query="technology",
            min_revenue=500000,
            min_team_size=20
        )
        print(f"   ✓ Found {len(combined)} results with combined search + filters")
        
        print("\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_validation_tests():
    """Test input validation"""
    print("\n=== VALIDATION TESTS ===")
    
    try:
        # Test invalid revenue
        try:
            create_company_profile(
                name="BadCorp",
                industry="Tech",
                location="USA",
                revenue=-1000,  # Invalid negative revenue
                team_size=10,
                founded=2020,
                website="https://bad.com",
                description="Bad company",
                needs="Nothing",
                challenges="Everything"
            )
            print("❌ Should have failed with negative revenue")
            return False
        except Exception as e:
            print(f"   ✓ Correctly rejected negative revenue: {type(e).__name__}")
        
        # Test invalid team size
        try:
            create_company_profile(
                name="BadCorp2",
                industry="Tech", 
                location="USA",
                revenue=100000,
                team_size=0,  # Invalid zero team size
                founded=2020,
                website="https://bad2.com",
                description="Bad company 2",
                needs="Nothing",
                challenges="Everything"
            )
            print("❌ Should have failed with zero team size")
            return False
        except Exception as e:
            print(f"   ✓ Correctly rejected zero team size: {type(e).__name__}")
            
        # Test empty search
        try:
            search_companies_by_text("")
            print("❌ Should have failed with empty search")
            return False
        except Exception as e:
            print(f"   ✓ Correctly rejected empty search: {type(e).__name__}")
            
        print("✅ VALIDATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ VALIDATION TEST FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting modular database system tests...\n")
    
    # Run tests
    basic_passed = run_basic_tests()
    validation_passed = run_validation_tests()
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS:")
    print(f"Basic functionality: {'✅ PASSED' if basic_passed else '❌ FAILED'}")
    print(f"Validation tests: {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    
    if basic_passed and validation_passed:
        print("\n🎉 ALL TESTS PASSED! The modular system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)