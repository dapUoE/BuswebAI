"""Service for all filtering operations on company data."""

from typing import Dict, List, Any, Optional, Union
from database_manager import DatabaseManager
from database_core import (
    get_revenue, get_team_size, get_founded, get_industry, 
    get_location, get_name, get_website
)


class FilterService:
    """Service for filtering companies based on various criteria."""
    
    def __init__(self, database_manager: DatabaseManager):
        """Initialize filter service with database dependency."""
        self.db = database_manager
    
    def filter_by_revenue_range(self, indices: List[int], min_revenue: Optional[int] = None, max_revenue: Optional[int] = None) -> List[int]:
        """Filter companies by revenue range."""
        if min_revenue is None and max_revenue is None:
            return indices
        
        filtered_indices = []
        for idx in indices:
            company = self.db.get_company(idx)
            if company:
                revenue = get_revenue(company)
                if min_revenue is not None and revenue < min_revenue:
                    continue
                if max_revenue is not None and revenue > max_revenue:
                    continue
                filtered_indices.append(idx)
        
        return filtered_indices
    
    def filter_by_team_size_range(self, indices: List[int], min_size: Optional[int] = None, max_size: Optional[int] = None) -> List[int]:
        """Filter companies by team size range."""
        if min_size is None and max_size is None:
            return indices
        
        filtered_indices = []
        for idx in indices:
            company = self.db.get_company(idx)
            if company:
                team_size = get_team_size(company)
                if min_size is not None and team_size < min_size:
                    continue
                if max_size is not None and team_size > max_size:
                    continue
                filtered_indices.append(idx)
        
        return filtered_indices
    
    def filter_by_founded_range(self, indices: List[int], min_year: Optional[int] = None, max_year: Optional[int] = None) -> List[int]:
        """Filter companies by founding year range."""
        if min_year is None and max_year is None:
            return indices
        
        filtered_indices = []
        for idx in indices:
            company = self.db.get_company(idx)
            if company:
                founded = get_founded(company)
                if min_year is not None and founded < min_year:
                    continue
                if max_year is not None and founded > max_year:
                    continue
                filtered_indices.append(idx)
        
        return filtered_indices
    
    def filter_by_industry(self, indices: List[int], industries: Union[str, List[str]]) -> List[int]:
        """Filter companies by industry (exact match, case-insensitive)."""
        if isinstance(industries, str):
            industries = [industries]
        
        industries_lower = [industry.lower().strip() for industry in industries]
        
        filtered_indices = []
        for idx in indices:
            company = self.db.get_company(idx)
            if company:
                company_industry = get_industry(company).lower().strip()
                if company_industry in industries_lower:
                    filtered_indices.append(idx)
        
        return filtered_indices
    
    def filter_by_location(self, indices: List[int], locations: Union[str, List[str]]) -> List[int]:
        """Filter companies by location (exact match, case-insensitive)."""
        if isinstance(locations, str):
            locations = [locations]
        
        locations_lower = [location.lower().strip() for location in locations]
        
        filtered_indices = []
        for idx in indices:
            company = self.db.get_company(idx)
            if company:
                company_location = get_location(company).lower().strip()
                if company_location in locations_lower:
                    filtered_indices.append(idx)
        
        return filtered_indices
    
    def filter_by_name_contains(self, indices: List[int], name_substring: str) -> List[int]:
        """Filter companies where name contains substring (case-insensitive)."""
        name_lower = name_substring.lower().strip()
        
        filtered_indices = []
        for idx in indices:
            company = self.db.get_company(idx)
            if company:
                company_name = get_name(company).lower()
                if name_lower in company_name:
                    filtered_indices.append(idx)
        
        return filtered_indices
    
    def filter_by_website_domain(self, indices: List[int], domain: str) -> List[int]:
        """Filter companies by website domain."""
        domain_lower = domain.lower().strip()
        
        filtered_indices = []
        for idx in indices:
            company = self.db.get_company(idx)
            if company:
                website = get_website(company).lower()
                if domain_lower in website:
                    filtered_indices.append(idx)
        
        return filtered_indices
    
    def apply_all_filters(self, indices: List[int], filters: Dict[str, Any]) -> List[int]:
        """Apply multiple filters in sequence."""
        current_indices = indices.copy()
        
        # Revenue filters
        if 'min_revenue' in filters or 'max_revenue' in filters:
            current_indices = self.filter_by_revenue_range(
                current_indices, 
                filters.get('min_revenue'), 
                filters.get('max_revenue')
            )
        
        # Team size filters
        if 'min_team_size' in filters or 'max_team_size' in filters:
            current_indices = self.filter_by_team_size_range(
                current_indices,
                filters.get('min_team_size'),
                filters.get('max_team_size')
            )
        
        # Founded year filters
        if 'min_founded' in filters or 'max_founded' in filters:
            current_indices = self.filter_by_founded_range(
                current_indices,
                filters.get('min_founded'),
                filters.get('max_founded')
            )
        
        # Industry filter
        if 'industry' in filters:
            current_indices = self.filter_by_industry(current_indices, filters['industry'])
        
        # Location filter
        if 'location' in filters:
            current_indices = self.filter_by_location(current_indices, filters['location'])
        
        # Name contains filter
        if 'name_contains' in filters:
            current_indices = self.filter_by_name_contains(current_indices, filters['name_contains'])
        
        # Website domain filter
        if 'website_domain' in filters:
            current_indices = self.filter_by_website_domain(current_indices, filters['website_domain'])
        
        return current_indices
    
    def get_all_company_indices(self) -> List[int]:
        """Get list of all valid company indices."""
        return self.db.get_all_indices()