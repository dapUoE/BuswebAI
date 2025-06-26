#!/usr/bin/env python3
"""
Script to add PE firms to the database with vector search capabilities.

Usage:
    python add_pe_firms.py

This script provides an interactive way to add PE firms to the database.
You can also modify the pe_firms_data list below to add your firms directly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pe_firm_service import PEFirmService, PEFirm

def main():
    """Main function to add PE firms"""
    
    # Example PE firms data - replace with your actual data
    pe_firms_data = [
        {
            "name": "Blackstone Group",
            "description": "One of the world's largest alternative asset managers, focused on private equity, real estate, hedge funds, and credit strategies. Known for large-scale buyouts and growth capital investments across various industries."
        },
        {
            "name": "KKR & Co.",
            "description": "Global investment firm specializing in private equity, energy, infrastructure, real estate, credit, and hedge funds. Focuses on management buyouts, leveraged buildups, and growth capital investments."
        },
        {
            "name": "Apollo Global Management",
            "description": "Leading global alternative investment manager with expertise in credit, private equity, and real assets. Known for distressed investments, corporate turnarounds, and complex financial situations."
        },
        {
            "name": "The Carlyle Group",
            "description": "Global investment firm with expertise across four business segments: Corporate Private Equity, Real Assets, Global Credit, and Investment Solutions. Focuses on middle-market and large-cap investments."
        },
        {
            "name": "TPG Inc.",
            "description": "Leading global alternative asset firm with investments in private equity, growth equity, impact investing, real estate, and public equity. Known for technology, healthcare, and consumer investments."
        }
    ]
    
    print("PE Firm Database Setup")
    print("=" * 50)
    
    try:
        # Initialize the service
        print("Initializing PE firm service...")
        pe_service = PEFirmService()
        
        # Check if user wants to use example data or input their own
        choice = input("\nWould you like to:\n1. Add example PE firms\n2. Add your own PE firms\n3. Search existing PE firms\n4. View all PE firms\n\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            # Add example firms
            print(f"\nAdding {len(pe_firms_data)} example PE firms...")
            added_ids = pe_service.add_pe_firms_from_list(pe_firms_data)
            print(f"Successfully added {len(added_ids)} PE firms!")
            
            # Show added firms
            print("\nAdded PE firms:")
            for i, firm_data in enumerate(pe_firms_data[:len(added_ids)]):
                print(f"  {i+1}. {firm_data['name']}")
        
        elif choice == "2":
            # Add custom firms
            print("\nEnter PE firm details (press Enter with empty name to finish):")
            added_count = 0
            
            while True:
                name = input("\nPE firm name: ").strip()
                if not name:
                    break
                
                description = input("Description: ").strip()
                if not description:
                    print("Description is required. Skipping this firm.")
                    continue
                
                try:
                    pe_firm = PEFirm(name=name, description=description)
                    firm_id = pe_service.add_pe_firm(pe_firm)
                    print(f"Added: {name} (ID: {firm_id})")
                    added_count += 1
                except Exception as e:
                    print(f"Error adding {name}: {e}")
            
            print(f"\nSuccessfully added {added_count} PE firms!")
        
        elif choice == "3":
            # Search PE firms
            query = input("\nEnter search query: ").strip()
            if query:
                print(f"\nSearching for: '{query}'")
                results = pe_service.search_pe_firms(query, top_k=5)
                
                if results:
                    print(f"\nFound {len(results)} matching PE firms:")
                    for i, (firm, score) in enumerate(results, 1):
                        print(f"\n{i}. {firm.name} (Score: {score})")
                        print(f"   Description: {firm.description[:100]}...")
                else:
                    print("No matching PE firms found.")
            else:
                print("No search query provided.")
        
        elif choice == "4":
            # View all firms
            firms = pe_service.get_all_pe_firms()
            if firms:
                print(f"\nAll PE firms ({len(firms)} total):")
                for i, firm in enumerate(firms, 1):
                    print(f"\n{i}. {firm.name}")
                    print(f"   Description: {firm.description[:100]}...")
            else:
                print("\nNo PE firms found in database.")
        
        else:
            print("Invalid choice.")
            return
        
        # Offer to test search functionality
        if choice in ["1", "2"] and input("\nWould you like to test the search functionality? (y/n): ").lower().startswith('y'):
            test_queries = [
                "technology investments",
                "healthcare focused",
                "distressed assets",
                "large buyouts",
                "growth capital"
            ]
            
            print("\nTesting search with sample queries:")
            for query in test_queries:
                print(f"\n--- Search: '{query}' ---")
                results = pe_service.search_pe_firms(query, top_k=3)
                
                if results:
                    for firm, score in results:
                        print(f"  • {firm.name} (Score: {score})")
                else:
                    print("  No results found")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    print("\nSetup complete!")
    return 0

if __name__ == "__main__":
    exit(main())