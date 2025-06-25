"""Main interface and example usage for the company database system."""

# from tabulate import tabulate  # Optional dependency
from typing import Dict, List, Any

# Import all modules
from config import DatabaseConfig
from models import Company, ValidationError, DatabaseError, CompanyNotFoundError
from validators import validate_company_data
from database_core import (
    append_to_db, get_db_length, get_last_index, is_valid_index, 
    get_item_by_index, copy_db, create_company_dict,
    get_name, get_description, get_needs, get_challenges, get_website,
    get_industry, get_location, get_revenue, get_team_size, get_founded
)
from embedding import create_embedding, combine_text_blob, add_embedding_to_index
from search import (
    company_db, vector_db, needs_vector_db, index, needs_index,
    search_companies_by_text, search_companies_by_needs, mark_indices_dirty
)
from filters import search_companies_with_filters, filter_companies

# === HIGH-LEVEL DATABASE FUNCTIONS ===
def add_company(company_data: Dict[str, Any]) -> int:
    """Add company to database and return index"""
    if not isinstance(company_data, dict):
        raise ValidationError("Company data must be a dictionary")
    
    # Validate all required fields are present
    for field in DatabaseConfig.REQUIRED_FIELDS:
        if field not in company_data:
            raise ValidationError(f"Missing required field: {field}")
    
    try:
        append_to_db(company_data, company_db)
        return get_last_index(company_db)
    except Exception as e:
        raise DatabaseError(f"Failed to add company: {str(e)}")

def get_company(index: int) -> Dict[str, Any]:
    """Get company by index"""
    try:
        from database_core import get_company as get_company_core
        return get_company_core(index, company_db)
    except Exception as e:
        raise DatabaseError(f"Failed to get company: {str(e)}")

def get_all_companies() -> List[Dict[str, Any]]:
    """Get all companies"""
    try:
        return copy_db(company_db)
    except Exception as e:
        raise DatabaseError(f"Failed to get all companies: {str(e)}")

def get_company_count() -> int:
    """Get total number of companies"""
    try:
        return get_db_length(company_db)
    except Exception as e:
        raise DatabaseError(f"Failed to get company count: {str(e)}")

def update_company(index: int, company_data: Dict[str, Any]) -> bool:
    """Update company at given index"""
    if not isinstance(index, int):
        raise ValidationError("Index must be an integer")
    
    if not isinstance(company_data, dict):
        raise ValidationError("Company data must be a dictionary")
    
    if not is_valid_index(index, company_db):
        raise CompanyNotFoundError(f"Company at index {index} not found")
    
    # Validate all required fields are present
    for field in DatabaseConfig.REQUIRED_FIELDS:
        if field not in company_data:
            raise ValidationError(f"Missing required field: {field}")
    
    try:
        # Update company data
        company_db[index] = company_data
        
        # Update corresponding embeddings
        desc_challenges_blob = combine_text_blob(
            get_description(company_data), 
            get_challenges(company_data)
        )
        new_desc_embedding = create_embedding(desc_challenges_blob)
        new_needs_embedding = create_embedding(get_needs(company_data))
        
        # Ensure vector databases are the right size
        while len(vector_db) <= index:
            vector_db.append(None)
        while len(needs_vector_db) <= index:
            needs_vector_db.append(None)
            
        vector_db[index] = new_desc_embedding
        needs_vector_db[index] = new_needs_embedding
        
        # Mark indices for rebuild
        mark_indices_dirty()
        
        return True
    except Exception as e:
        raise DatabaseError(f"Failed to update company: {str(e)}")

def delete_company(index: int) -> bool:
    """Delete company at given index"""
    if not isinstance(index, int):
        raise ValidationError("Index must be an integer")
    
    if not is_valid_index(index, company_db):
        raise CompanyNotFoundError(f"Company at index {index} not found")
    
    try:
        # Remove from all databases
        company_db.pop(index)
        
        # Remove from vector databases if they exist at this index
        if index < len(vector_db):
            vector_db.pop(index)
        if index < len(needs_vector_db):
            needs_vector_db.pop(index)
        
        # Mark indices for rebuild
        mark_indices_dirty()
        
        return True
    except Exception as e:
        raise DatabaseError(f"Failed to delete company: {str(e)}")

def create_company_profile(name: str, industry: str, location: str, revenue: int, 
                          team_size: int, founded: int, website: str, 
                          description: str, needs: str, challenges: str) -> int:
    """Create a complete company profile with validation"""
    try:
        # Validate and create company data
        company_data = create_company_dict(name, industry, location, revenue, 
                                         team_size, founded, website, 
                                         description, needs, challenges)
        
        # Create separate embeddings
        desc_challenges_blob = combine_text_blob(description, challenges)
        desc_embedding = create_embedding(desc_challenges_blob)
        needs_embedding = create_embedding(needs)
        
        # Add to databases
        add_embedding_to_index(desc_embedding, vector_db, index, "main")
        add_embedding_to_index(needs_embedding, needs_vector_db, needs_index, "needs")
        return add_company(company_data)
    except Exception as e:
        raise DatabaseError(f"Failed to create company profile: {str(e)}")

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    try:
        # Create sample companies
        create_company_profile(
            name="FinNest",
            industry="FinTech",
            location="UK",
            revenue=1_200_000,
            team_size=15,
            founded=2021,
            website="https://finnest.com",
            description="A platform helping Gen Z save and invest.",
            needs="Looking for banking partnerships and mobile app collaborators.",
            challenges="Struggling to engage users in Tier 2 cities."
        )

        create_company_profile(
            name="HealthAI",
            industry="HealthTech",
            location="Germany",
            revenue=800_000,
            team_size=10,
            founded=2020,
            website="https://healthai.org",
            description="Uses AI to detect disease early through wearable data.",
            needs="Seeking research hospitals and AI funding partners.",
            challenges="Data privacy regulations limit pilot tests."
        )

        # Add more sample companies for filtering demos
        create_company_profile(
            name="StartupAI",
            industry="AI/ML", 
            location="USA",
            revenue=500_000,
            team_size=8,
            founded=2022,
            website="https://startupai.com",
            description="Early-stage AI startup for small businesses",
            needs="Seeking seed funding and technical mentors",
            challenges="Finding product-market fit and scaling team"
        )

        create_company_profile(
            name="MegaCorp Solutions",
            industry="Enterprise Software",
            location="Germany", 
            revenue=50_000_000,
            team_size=200,
            founded=2015,
            website="https://megacorp.de",
            description="Large enterprise software solutions provider",
            needs="Expanding into new international markets",
            challenges="Legacy system modernization and competition"
        )

        # Create 3 more very different companies
        create_company_profile(
            name="EcoTech Innovations",
            industry="Sustainability",
            location="Canada",
            revenue=2_500_000,
            team_size=30,
            founded=2018,
            website="https://ecotech.ca",
            description="Developing green technologies for urban areas",
            needs="Partnerships with local governments and NGOs",
            challenges="Regulatory hurdles and public awareness"
        )

        create_company_profile(
            name="EduFuture",
            industry="EdTech",
            location="USA",
            revenue=1_000_000,
            team_size=20,
            founded=2019,
            website="https://edufuture.com",
            description="Online learning platform for K-12 students",
            needs="Content creators and educational partnerships",
            challenges="Adapting to diverse learning styles"
        )

        create_company_profile(
            name="TravelSmart",
            industry="TravelTech",
            location="UK",
            revenue=3_000_000,
            team_size=50,
            founded=2020,
            website="https://travelsmart.co.uk",
            description="AI-powered travel planning and booking service",
            needs="Integration with airlines and hotels",
            challenges="High competition and customer retention"
        )

        print("=== SEMANTIC SEARCH (Description + Challenges) ===")
        results = search_companies_by_text("AI-based health partners")
        for r in results:
            print(f"{r['name']} -> {r['match_score']} - {r['description']}")

        print("\n=== NEEDS-SPECIFIC SEARCH ===")
        needs_results = search_companies_by_needs("partnerships funding research")
        for r in needs_results:
            print(f"{r['name']} -> {r['match_score']} - {r['needs']}")

        print("\n=== FILTERING EXAMPLES ===")
        
        # Example 1: Filter by industry
        fintech_companies = filter_companies(industry="FinTech")
        print(f"\nFinTech companies: {len(fintech_companies)}")
        for company in fintech_companies:
            print(f"  - {company['name']} ({company['location']}, ${company['revenue']:,})")

        # Example 2: Filter by revenue range
        high_revenue = filter_companies(min_revenue=1_000_000)
        print(f"\nCompanies with >$1M revenue: {len(high_revenue)}")
        for company in high_revenue:
            print(f"  - {company['name']}: ${company['revenue']:,}")

        # Example 3: Stackable filters - HealthTech in Germany with <1000 employees
        german_healthtech = filter_companies(
            industry="HealthTech", 
            location="Germany", 
            max_team_size=100
        )
        print(f"\nGerman HealthTech companies (<100 employees): {len(german_healthtech)}")
        for company in german_healthtech:
            print(f"  - {company['name']}: {company['team_size']} employees")

        # Example 4: Combined semantic search + filters
        ai_startups = search_companies_with_filters(
            text_query="AI startup funding",
            min_founded=2020,
            max_revenue=5_000_000,
            industry=["AI/ML", "HealthTech"]
        )
        print(f"\nAI startups (founded after 2020, <$5M revenue): {len(ai_startups)}")
        for company in ai_startups:
            score_text = f"score: {company['match_score']}" if company['match_score'] else "no score"
            print(f"  - {company['name']} ({score_text})")

        # Example 5: Multiple locations and industries
        uk_us_tech = filter_companies(
            location=["UK", "USA"],
            industry=["AI/ML", "FinTech"],
            min_team_size=5
        )
        print(f"\nUK/USA Tech companies (>5 employees): {len(uk_us_tech)}")
        for company in uk_us_tech:
            print(f"  - {company['name']} ({company['location']}, {company['industry']})")

        print("\n" + "="*50)
        print("FULL RESULTS TABLE:")
        # print(tabulate(results, headers="keys"))  # Requires tabulate package
        for i, result in enumerate(results):
            print(f"{i+1}. {result['name']} (Score: {result['match_score']})")
            print(f"   Description: {result['description']}")
            print(f"   Needs: {result['needs']}")
            print()

        # === Example 6: Match company needs against other companies' descriptions ===
        print("\n" + "="*60)
        print("🔍 EXAMPLE 6: NEEDS MATCHING")
        needs_query = "we need some help growing our small business"
        print(f"Searching for companies matching the need:\n  → \"{needs_query}\"\n")

        try:
            matches = search_companies_by_text(needs_query)
            print(f"Found {len(matches)} matching companies:\n")
            for match in matches:
                print(f"- {match['name']} (Score: {match['match_score']})")
                print(f"  Description: {match['description']}\n")

        except ValidationError as ve:
            print(f"[Validation Error] {str(ve)}")
        except DatabaseError as de:
            print(f"[Database Error] {str(de)}")
        except CompanyNotFoundError as cnfe:
            print(f"[Company Not Found] {str(cnfe)}")

            # === Example 7: Find companies offering AI expertise and mentoring ===
        print("\n" + "="*60)
        print("🔍 EXAMPLE 7: NEEDS MATCHING FOR TECHNICAL SUPPORT")
        needs_query = "we are looking for travel expertise and local mentors"
        print(f"Searching for companies matching the need:\n  → \"{needs_query}\"\n")

        try:
            matches = search_companies_by_text(needs_query)
            print(f"Found {len(matches)} matching companies:\n")
            for match in matches:
                print(f"- {match['name']} (Score: {match['match_score']})")
                print(f"  Description: {match['description']}\n")

        except ValidationError as ve:
            print(f"[Validation Error] {str(ve)}")
        except DatabaseError as de:
            print(f"[Database Error] {str(de)}")
        except CompanyNotFoundError as cnfe:
            print(f"[Company Not Found] {str(cnfe)}")

     
        
    except Exception as e:
        print(f"Error: {str(e)}")

