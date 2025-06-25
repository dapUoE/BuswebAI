"""Service for all search operations and result formatting."""

from typing import Dict, List, Any, Optional
from database_manager import DatabaseManager
from embedding_service import EmbeddingService
from database_core import get_name, get_description, get_needs, get_challenges, get_website
from models import DatabaseError
from config import DatabaseConfig


class SearchService:
    """Service for managing search operations and result formatting."""
    
    def __init__(self, database_manager: DatabaseManager, embedding_service: EmbeddingService):
        """Initialize search service with dependencies."""
        self.db = database_manager
        self.embedding = embedding_service
    
    def search_by_description(self, query: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Search companies by description + challenges text."""
        try:
            # Get search results from FAISS
            distances, indices = self.embedding.search_index(
                query, self.db.get_main_index(), top_k
            )
            
            # Format results
            results = []
            for idx, score in zip(indices, distances):
                idx = int(idx)
                score = float(score)
                company = self.db.get_company(idx)
                
                if company:
                    result = self._create_search_result(company, score)
                    results.append(result)
            
            return results
        except Exception as e:
            raise DatabaseError(f"Failed to search by description: {str(e)}")
    
    def search_by_needs(self, query: str, top_k: int = DatabaseConfig.DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Search companies by their needs specifically."""
        try:
            # Get search results from needs FAISS index
            distances, indices = self.embedding.search_index(
                query, self.db.get_needs_index(), top_k
            )
            
            # Format results
            results = []
            for idx, score in zip(indices, distances):
                idx = int(idx)
                score = float(score)
                company = self.db.get_company(idx)
                
                if company:
                    result = self._create_search_result(company, score)
                    results.append(result)
            
            return results
        except Exception as e:
            raise DatabaseError(f"Failed to search by needs: {str(e)}")
    
    def _create_search_result(self, company: Dict[str, Any], score: float) -> Dict[str, Any]:
        """Create standardized search result from company data."""
        try:
            return {
                "name": get_name(company),
                "match_score": self.embedding.round_score(score),
                "description": get_description(company),
                "needs": get_needs(company),
                "challenges": get_challenges(company),
                "website": get_website(company)
            }
        except Exception as e:
            raise DatabaseError(f"Failed to create search result: {str(e)}")
    
    def get_companies_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get companies by their indices (for filtered results)."""
        try:
            companies = []
            for idx in indices:
                company = self.db.get_company(idx)
                if company:
                    companies.append(company)
            return companies
        except Exception as e:
            raise DatabaseError(f"Failed to get companies by indices: {str(e)}")
    
    def format_filtered_results(self, companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format filtered companies into standardized result format."""
        try:
            results = []
            for company in companies:
                result = {
                    "name": get_name(company),
                    "match_score": None,  # No semantic matching for pure filters
                    "description": get_description(company),
                    "needs": get_needs(company),
                    "challenges": get_challenges(company),
                    "website": get_website(company),
                    "industry": company.get("industry"),
                    "location": company.get("location"),
                    "revenue": company.get("revenue"),
                    "team_size": company.get("team_size"),
                    "founded": company.get("founded")
                }
                results.append(result)
            return results
        except Exception as e:
            raise DatabaseError(f"Failed to format filtered results: {str(e)}")
    
    def search_with_semantic_and_filters(self, query: str, filtered_indices: List[int], top_k: int) -> List[Dict[str, Any]]:
        """Perform semantic search on pre-filtered company indices."""
        try:
            # Get semantic search results on all companies first
            semantic_results = self.search_by_description(query, top_k=min(top_k * 3, self.db.get_company_count()))
            
            # Filter semantic results to only include filtered indices
            filtered_results = []
            for i, result in enumerate(semantic_results):
                if i in filtered_indices[:top_k]:
                    filtered_results.append(result)
            
            return filtered_results
        except Exception as e:
            raise DatabaseError(f"Failed to search with semantic and filters: {str(e)}")