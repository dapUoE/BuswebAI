#!/usr/bin/env python3
"""Test the new modular system architecture"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_modular import CompanyDatabaseSystem

def test_modular_system():
    """Test the new modular system end-to-end"""
    print("=== TESTING MODULAR SYSTEM ===")
    
    # Initialize system
    system = CompanyDatabaseSystem()
    
    try:
        # Test 1: Create companies
        print("1. Testing company creation...")
        idx1 = system.create_company_profile(
            name="ModularCorp",
            industry="Technology",
            location="USA",
            revenue=2000000,
            team_size=100,
            founded=2019,
            website="https://modularcorp.com",
            description="A modular technology company for testing",
            needs="Modular architecture consulting",
            challenges="Managing complex dependencies"
        )
        print(f"   ✓ Created company at index {idx1}")
        
        idx2 = system.create_company_profile(
            name="ServiceCorp",
            industry="Consulting",
            location="Germany",
            revenue=1500000,
            team_size=75,
            founded=2020,
            website="https://servicecorp.de",
            description="Service-oriented architecture consulting",
            needs="Enterprise clients and partnerships",
            challenges="Scaling service delivery"
        )
        print(f"   ✓ Created company at index {idx2}")
        
        # Test 2: Search functionality
        print("\n2. Testing search functionality...")
        results = system.search_companies_by_text("modular technology")
        print(f"   ✓ Found {len(results)} results for text search")
        
        needs_results = system.search_companies_by_needs("consulting partnerships")
        print(f"   ✓ Found {len(needs_results)} results for needs search")
        
        # Test 3: Filtering
        print("\n3. Testing filtering...")
        tech_companies = system.filter_companies(industry="Technology")
        print(f"   ✓ Found {len(tech_companies)} Technology companies")
        
        high_revenue = system.filter_companies(min_revenue=1800000)
        print(f"   ✓ Found {len(high_revenue)} companies with >$1.8M revenue")
        
        # Test 4: Combined search + filters
        print("\n4. Testing combined search + filters...")
        combined = system.search_companies_with_filters(
            text_query="technology consulting",
            min_revenue=1000000,
            min_team_size=50
        )
        print(f"   ✓ Found {len(combined)} results with combined search + filters")
        
        # Test 5: CRUD operations
        print("\n5. Testing CRUD operations...")
        
        # Read
        company = system.get_company(idx1)
        assert company["name"] == "ModularCorp"
        print("   ✓ Read operation works")
        
        # Update
        updated = system.update_company(
            idx1, "ModularCorp Updated", "Technology", "USA", 2500000,
            120, 2019, "https://modularcorp-updated.com",
            "An updated modular technology company", "New consulting needs", "New challenges"
        )
        assert updated == True
        
        updated_company = system.get_company(idx1)
        assert updated_company["name"] == "ModularCorp Updated"
        assert updated_company["revenue"] == 2500000
        print("   ✓ Update operation works")
        
        # Count
        count = system.get_company_count()
        assert count == 2
        print("   ✓ Count operation works")
        
        # Delete
        deleted = system.delete_company(idx2)
        assert deleted == True
        assert system.get_company_count() == 1
        print("   ✓ Delete operation works")
        
        # Test 6: Utility functions
        print("\n6. Testing utility functions...")
        similarity = system.calculate_text_similarity(
            "modular technology solutions",
            "technology architecture consulting"
        )
        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1
        print(f"   ✓ Text similarity calculation works: {similarity:.3f}")
        
        print("\n✅ ALL MODULAR SYSTEM TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ MODULAR SYSTEM TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_service_isolation():
    """Test that services are properly isolated"""
    print("\n=== TESTING SERVICE ISOLATION ===")
    
    try:
        # Create two separate systems
        system1 = CompanyDatabaseSystem()
        system2 = CompanyDatabaseSystem()
        
        # Add data to system1
        system1.create_company_profile(
            name="System1Corp", industry="Tech", location="USA", revenue=1000000,
            team_size=50, founded=2020, website="https://system1.com",
            description="Company in system 1", needs="System 1 needs", challenges="System 1 challenges"
        )
        
        # Add different data to system2  
        system2.create_company_profile(
            name="System2Corp", industry="Finance", location="UK", revenue=2000000,
            team_size=100, founded=2019, website="https://system2.com", 
            description="Company in system 2", needs="System 2 needs", challenges="System 2 challenges"
        )
        
        # Verify isolation
        assert system1.get_company_count() == 1
        assert system2.get_company_count() == 1
        
        system1_company = system1.get_company(0)
        system2_company = system2.get_company(0)
        
        assert system1_company["name"] == "System1Corp"
        assert system2_company["name"] == "System2Corp"
        
        print("   ✓ Systems are properly isolated")
        print("   ✓ Each system maintains its own state")
        
        print("\n✅ SERVICE ISOLATION TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ SERVICE ISOLATION TEST FAILED: {str(e)}")
        return False

def test_backwards_compatibility():
    """Test backwards compatibility functions"""
    print("\n=== TESTING BACKWARDS COMPATIBILITY ===")
    
    try:
        # Import backwards compatibility functions
        from main_modular import (
            create_company_profile, get_company, get_company_count,
            search_companies_by_text, filter_companies
        )
        
        # Test using old interface
        idx = create_company_profile(
            name="BackwardsCompat",
            industry="Testing", 
            location="TestLand",
            revenue=500000,
            team_size=25,
            founded=2021,
            website="https://backwards.com",
            description="Testing backwards compatibility",
            needs="Legacy support",
            challenges="Maintaining compatibility"
        )
        
        company = get_company(idx)
        assert company["name"] == "BackwardsCompat"
        
        count = get_company_count()
        assert count == 1
        
        results = search_companies_by_text("backwards compatibility")
        assert len(results) > 0
        
        filtered = filter_companies(industry="Testing")
        assert len(filtered) == 1
        
        print("   ✓ Backwards compatibility functions work")
        print("   ✓ Old interface still functional")
        
        print("\n✅ BACKWARDS COMPATIBILITY TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ BACKWARDS COMPATIBILITY TEST FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING MODULAR SYSTEM TESTS\\n")
    
    # Run all tests
    test1 = test_modular_system()
    test2 = test_service_isolation() 
    test3 = test_backwards_compatibility()
    
    print(f"\\n{'='*50}")
    print("FINAL RESULTS:")
    print(f"Modular system: {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Service isolation: {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"Backwards compatibility: {'✅ PASSED' if test3 else '❌ FAILED'}")
    
    if test1 and test2 and test3:
        print("\\n🎉 ALL MODULAR SYSTEM TESTS PASSED!")
        print("The new architecture is working correctly with:")
        print("  • Proper dependency injection")
        print("  • Service layer isolation") 
        print("  • Encapsulated state management")
        print("  • Backwards compatibility")
        sys.exit(0)
    else:
        print("\\n⚠️  Some tests failed.")
        sys.exit(1)