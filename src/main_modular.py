"""Modular main interface using dependency injection and service layer architecture."""

from typing import Dict, List, Any, Optional
from database_manager import DatabaseManager
from embedding_service import EmbeddingService
from company_service import CompanyService
from config import DatabaseConfig


class CompanyDatabaseSystem:
    """Main system class that coordinates all services."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the system with dependency injection."""
        # Initialize core services
        self.db_manager = DatabaseManager()
        self.embedding_service = EmbeddingService(model_name)
        self.company_service = CompanyService(self.db_manager, self.embedding_service)
    
    # === HIGH-LEVEL COMPANY OPERATIONS ===
    
    def create_company_profile(self, name: str, industry: str, location: str, revenue: int,
                              team_size: int, founded: int, website: str,
                              description: str, needs: str, challenges: str) -> int:
        """Create a complete company profile."""
        return self.company_service.create_company_profile(
            name, industry, location, revenue, team_size,
            founded, website, description, needs, challenges
        )
    
    def update_company(self, index: int, name: str, industry: str, location: str, revenue: int,
                      team_size: int, founded: int, website: str,
                      description: str, needs: str, challenges: str) -> bool:
        """Update company profile."""
        return self.company_service.update_company_profile(
            index, name, industry, location, revenue, team_size,
            founded, website, description, needs, challenges
        )
    
    def delete_company(self, index: int) -> bool:
        """Delete company profile."""
        return self.company_service.delete_company_profile(index)
    
    def get_company(self, index: int) -> Optional[Dict[str, Any]]:
        """Get company by index."""
        return self.company_service.get_company_profile(index)
    
    def get_all_companies(self) -> List[Dict[str, Any]]:
        """Get all companies."""
        return self.company_service.get_all_company_profiles()
    
    def get_company_count(self) -> int:
        """Get total number of companies."""
        return self.company_service.get_company_count()
    
    # === SEARCH OPERATIONS ===
    
    def search_companies_by_text(self, query: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Search companies by description text."""
        return self.company_service.search_companies_by_text(query, top_k)
    
    def search_companies_by_needs(self, query: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Search companies by their needs."""
        return self.company_service.search_companies_by_needs(query, top_k)
    
    # === FILTERING OPERATIONS ===
    
    def filter_companies(self, **filters) -> List[Dict[str, Any]]:
        """Filter companies by various criteria."""
        return self.company_service.filter_companies(**filters)
    
    def search_companies_with_filters(self, text_query: Optional[str] = None, 
                                     top_k: int = DatabaseConfig.DEFAULT_TOP_K, 
                                     **filters) -> List[Dict[str, Any]]:
        """Advanced search combining text query and filters."""
        return self.company_service.search_with_filters(text_query, top_k, **filters)
    
    # === UTILITY OPERATIONS ===
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return self.embedding_service.calculate_similarity(text1, text2)
    
    def clear_all_data(self):
        """Clear all data (useful for testing)."""
        self.db_manager.clear_all_data()


# === BACKWARDS COMPATIBILITY FUNCTIONS ===
# These provide the same interface as the old main.py for existing code

# Global system instance (lazy-loaded)
_system: Optional[CompanyDatabaseSystem] = None

def _get_system() -> CompanyDatabaseSystem:
    """Get or create global system instance."""
    global _system
    if _system is None:
        _system = CompanyDatabaseSystem()
    return _system

def create_company_profile(name: str, industry: str, location: str, revenue: int,
                          team_size: int, founded: int, website: str,
                          description: str, needs: str, challenges: str) -> int:
    """Create a complete company profile."""
    return _get_system().create_company_profile(
        name, industry, location, revenue, team_size,
        founded, website, description, needs, challenges
    )

def update_company(index: int, company_data: Dict[str, Any]) -> bool:
    """Update company profile (backwards compatible)."""
    system = _get_system()
    return system.update_company(
        index,
        company_data['name'], company_data['industry'], company_data['location'],
        company_data['revenue'], company_data['team_size'], company_data['founded'],
        company_data['website'], company_data['description'], 
        company_data['needs'], company_data['challenges']
    )

def delete_company(index: int) -> bool:
    """Delete company profile."""
    return _get_system().delete_company(index)

def get_company(index: int) -> Optional[Dict[str, Any]]:
    """Get company by index."""
    return _get_system().get_company(index)

def get_all_companies() -> List[Dict[str, Any]]:
    """Get all companies."""
    return _get_system().get_all_companies()

def get_company_count() -> int:
    """Get total number of companies."""
    return _get_system().get_company_count()

def add_company(company_data: Dict[str, Any]) -> int:
    """Add company (backwards compatible - minimal validation)."""
    system = _get_system()
    return system.db_manager.add_company(company_data)

def search_companies_by_text(query: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Search companies by text."""
    return _get_system().search_companies_by_text(query, top_k)

def search_companies_by_needs(query: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Search companies by needs."""
    return _get_system().search_companies_by_needs(query, top_k)

def filter_companies(**filters) -> List[Dict[str, Any]]:
    """Filter companies."""
    return _get_system().filter_companies(**filters)

def search_companies_with_filters(text_query: Optional[str] = None, 
                                 top_k: int = DatabaseConfig.DEFAULT_TOP_K, 
                                 **filters) -> List[Dict[str, Any]]:
    """Search with filters."""
    return _get_system().search_companies_with_filters(text_query, top_k, **filters)


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    # Initialize system
    system = CompanyDatabaseSystem()
    
    try:
        # Create sample companies
        system.create_company_profile(
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

        system.create_company_profile(
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

        print("=== SEMANTIC SEARCH (Description + Challenges) ===\")
        results = system.search_companies_by_text("AI-based health partners")
        for r in results:
            print(f"{r['name']} -> {r['match_score']} - {r['description']}")

        print("\\n=== NEEDS-SPECIFIC SEARCH ===\")
        needs_results = system.search_companies_by_needs("partnerships funding research")
        for r in needs_results:
            print(f"{r['name']} -> {r['match_score']} - {r['needs']}")

        print("\\n=== FILTERING EXAMPLES ===\")
        
        # Filter by industry
        fintech_companies = system.filter_companies(industry="FinTech")
        print(f"\\nFinTech companies: {len(fintech_companies)}")
        for company in fintech_companies:
            print(f"  - {company['name']} ({company['location']}, ${company['revenue']:,})")

        # Combined search + filters
        ai_startups = system.search_companies_with_filters(
            text_query="AI startup funding",
            min_founded=2020,
            max_revenue=5_000_000
        )
        print(f"\\nAI startups (founded after 2020, <$5M revenue): {len(ai_startups)}")
        for company in ai_startups:
            score_text = f"score: {company['match_score']}" if company['match_score'] else "no score"
            print(f"  - {company['name']} ({score_text})")
        
        print("\\n✅ Modular system working correctly!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()