"""Service for company-related business logic and operations."""

from typing import Dict, List, Any
from database_manager import DatabaseManager
from embedding_service import EmbeddingService
from search_service import SearchService
from filter_service import FilterService
from validators import validate_company_data
from database_core import get_description, get_challenges, get_needs
from models import DatabaseError


class CompanyService:
    """Service for high-level company operations combining multiple services."""
    
    def __init__(self, database_manager: DatabaseManager, embedding_service: EmbeddingService):
        """Initialize company service with dependencies."""
        self.db = database_manager
        self.embedding = embedding_service
        self.search = SearchService(database_manager, embedding_service)
        self.filter = FilterService(database_manager)
    
    def create_company_profile(self, name: str, industry: str, location: str, revenue: int, 
                              team_size: int, founded: int, website: str, 
                              description: str, needs: str, challenges: str) -> int:
        """Create a complete company profile with validation and embeddings."""
        try:
            # Step 1: Validate and create company data
            company_data = self._validate_company_data(
                name, industry, location, revenue, team_size, 
                founded, website, description, needs, challenges
            )
            
            # Step 2: Create embeddings
            desc_embedding, needs_embedding = self._create_company_embeddings(
                description, challenges, needs
            )
            
            # Step 3: Persist all data
            company_idx = self._persist_company_data(company_data, desc_embedding, needs_embedding)
            
            return company_idx
        except Exception as e:
            raise DatabaseError(f"Failed to create company profile: {str(e)}")
    
    def update_company_profile(self, index: int, name: str, industry: str, location: str, revenue: int,
                              team_size: int, founded: int, website: str,
                              description: str, needs: str, challenges: str) -> bool:
        """Update a complete company profile with validation and embeddings."""
        try:
            # Step 1: Validate new company data
            company_data = self._validate_company_data(
                name, industry, location, revenue, team_size,
                founded, website, description, needs, challenges
            )
            
            # Step 2: Update company in database
            self.db.update_company(index, company_data)
            
            # Step 3: Create new embeddings
            desc_embedding, needs_embedding = self._create_company_embeddings(
                description, challenges, needs
            )
            
            # Step 4: Update embeddings
            self.db.update_embeddings(index, desc_embedding, needs_embedding)
            
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to update company profile: {str(e)}")
    
    def delete_company_profile(self, index: int) -> bool:
        """Delete a complete company profile."""
        return self.db.delete_company(index)
    
    def get_company_profile(self, index: int) -> Dict[str, Any]:
        """Get company profile by index."""
        return self.db.get_company(index)
    
    def get_all_company_profiles(self) -> List[Dict[str, Any]]:
        """Get all company profiles."""
        return self.db.get_all_companies()
    
    def get_company_count(self) -> int:
        """Get total number of companies."""
        return self.db.get_company_count()
    
    def search_companies_by_text(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search companies by description text."""
        return self.search.search_by_description(query, top_k)
    
    def search_companies_by_needs(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search companies by their needs."""
        return self.search.search_by_needs(query, top_k)
    
    def filter_companies(self, **filters) -> List[Dict[str, Any]]:
        """Filter companies by various criteria."""
        all_indices = self.filter.get_all_company_indices()
        filtered_indices = self.filter.apply_all_filters(all_indices, filters)
        
        # Get companies and format results
        companies = self.search.get_companies_by_indices(filtered_indices)
        return self.search.format_filtered_results(companies)
    
    def search_with_filters(self, text_query: str = None, top_k: int = 3, **filters) -> List[Dict[str, Any]]:
        """Advanced search combining text query and filters."""
        try:
            # Get all indices and apply filters
            all_indices = self.filter.get_all_company_indices()
            filtered_indices = self.filter.apply_all_filters(all_indices, filters)
            
            if text_query and text_query.strip():
                # Semantic search + filters
                return self.search.search_with_semantic_and_filters(text_query, filtered_indices, top_k)
            else:
                # Pure filtering
                companies = self.search.get_companies_by_indices(filtered_indices[:top_k])
                return self.search.format_filtered_results(companies)
                
        except Exception as e:
            raise DatabaseError(f"Failed to search with filters: {str(e)}")
    
    # === PRIVATE HELPER METHODS ===
    
    def _validate_company_data(self, name: str, industry: str, location: str, revenue: int,
                              team_size: int, founded: int, website: str,
                              description: str, needs: str, challenges: str) -> Dict[str, Any]:
        """Validate and clean company data."""
        return validate_company_data(
            name, industry, location, revenue, team_size,
            founded, website, description, needs, challenges
        )
    
    def _create_company_embeddings(self, description: str, challenges: str, needs: str) -> tuple:
        """Create both description and needs embeddings."""
        desc_embedding = self.embedding.create_description_embedding(description, challenges)
        needs_embedding = self.embedding.create_needs_embedding(needs)
        return desc_embedding, needs_embedding
    
    def _persist_company_data(self, company_data: Dict[str, Any], desc_embedding, needs_embedding) -> int:
        """Persist company data and embeddings to databases."""
        # Add embeddings to vector databases
        self.db.add_embedding(desc_embedding)
        self.db.add_needs_embedding(needs_embedding)
        
        # Add company to main database
        return self.db.add_company(company_data)