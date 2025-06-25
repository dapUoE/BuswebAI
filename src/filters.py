"""Filtering functionality for the company database system."""

from typing import Dict, List, Any, Optional, Union

from config import DatabaseConfig
from models import ValidationError, DatabaseError
from database_core import (
    get_company, get_name, get_industry, get_location, get_revenue, 
    get_team_size, get_founded, get_website, get_description, 
    get_needs, get_challenges, get_db_length
)
from search import search_companies_by_text, company_db

# === INDIVIDUAL FILTER FUNCTIONS ===
def filter_by_revenue_range(company_indices: List[int], min_revenue: Optional[int] = None, max_revenue: Optional[int] = None) -> List[int]:
    """Filter companies by revenue range"""
    if min_revenue is None and max_revenue is None:
        return company_indices
    
    filtered_indices = []
    for idx in company_indices:
        company = get_company(idx, company_db)
        if company:
            revenue = get_revenue(company)
            if min_revenue is not None and revenue < min_revenue:
                continue
            if max_revenue is not None and revenue > max_revenue:
                continue
            filtered_indices.append(idx)
    
    return filtered_indices

def filter_by_team_size_range(company_indices: List[int], min_size: Optional[int] = None, max_size: Optional[int] = None) -> List[int]:
    """Filter companies by team size range"""
    if min_size is None and max_size is None:
        return company_indices
    
    filtered_indices = []
    for idx in company_indices:
        company = get_company(idx, company_db)
        if company:
            team_size = get_team_size(company)
            if min_size is not None and team_size < min_size:
                continue
            if max_size is not None and team_size > max_size:
                continue
            filtered_indices.append(idx)
    
    return filtered_indices

def filter_by_founded_range(company_indices: List[int], min_year: Optional[int] = None, max_year: Optional[int] = None) -> List[int]:
    """Filter companies by founding year range"""
    if min_year is None and max_year is None:
        return company_indices
    
    filtered_indices = []
    for idx in company_indices:
        company = get_company(idx, company_db)
        if company:
            founded = get_founded(company)
            if min_year is not None and founded < min_year:
                continue
            if max_year is not None and founded > max_year:
                continue
            filtered_indices.append(idx)
    
    return filtered_indices

def filter_by_industry(company_indices: List[int], industries: Union[str, List[str]]) -> List[int]:
    """Filter companies by industry (exact match, case-insensitive)"""
    if isinstance(industries, str):
        industries = [industries]
    
    industries_lower = [industry.lower().strip() for industry in industries]
    
    filtered_indices = []
    for idx in company_indices:
        company = get_company(idx, company_db)
        if company:
            company_industry = get_industry(company).lower().strip()
            if company_industry in industries_lower:
                filtered_indices.append(idx)
    
    return filtered_indices

def filter_by_location(company_indices: List[int], locations: Union[str, List[str]]) -> List[int]:
    """Filter companies by location (exact match, case-insensitive)"""
    if isinstance(locations, str):
        locations = [locations]
    
    locations_lower = [location.lower().strip() for location in locations]
    
    filtered_indices = []
    for idx in company_indices:
        company = get_company(idx, company_db)
        if company:
            company_location = get_location(company).lower().strip()
            if company_location in locations_lower:
                filtered_indices.append(idx)
    
    return filtered_indices

def filter_by_name_contains(company_indices: List[int], name_substring: str) -> List[int]:
    """Filter companies where name contains substring (case-insensitive)"""
    name_lower = name_substring.lower().strip()
    
    filtered_indices = []
    for idx in company_indices:
        company = get_company(idx, company_db)
        if company:
            company_name = get_name(company).lower()
            if name_lower in company_name:
                filtered_indices.append(idx)
    
    return filtered_indices

def filter_by_website_domain(company_indices: List[int], domain: str) -> List[int]:
    """Filter companies by website domain"""
    domain_lower = domain.lower().strip()
    
    filtered_indices = []
    for idx in company_indices:
        company = get_company(idx, company_db)
        if company:
            website = get_website(company).lower()
            if domain_lower in website:
                filtered_indices.append(idx)
    
    return filtered_indices

# === FILTER COMBINATION FUNCTIONS ===
def get_all_company_indices() -> List[int]:
    """Get list of all valid company indices"""
    return list(range(get_db_length(company_db)))

def apply_filters(company_indices: List[int], filters: Dict[str, Any]) -> List[int]:
    """Apply multiple filters in sequence"""
    current_indices = company_indices.copy()
    
    # Revenue filters
    if 'min_revenue' in filters or 'max_revenue' in filters:
        current_indices = filter_by_revenue_range(
            current_indices, 
            filters.get('min_revenue'), 
            filters.get('max_revenue')
        )
    
    # Team size filters
    if 'min_team_size' in filters or 'max_team_size' in filters:
        current_indices = filter_by_team_size_range(
            current_indices,
            filters.get('min_team_size'),
            filters.get('max_team_size')
        )
    
    # Founded year filters
    if 'min_founded' in filters or 'max_founded' in filters:
        current_indices = filter_by_founded_range(
            current_indices,
            filters.get('min_founded'),
            filters.get('max_founded')
        )
    
    # Industry filter
    if 'industry' in filters:
        current_indices = filter_by_industry(current_indices, filters['industry'])
    
    # Location filter
    if 'location' in filters:
        current_indices = filter_by_location(current_indices, filters['location'])
    
    # Name contains filter
    if 'name_contains' in filters:
        current_indices = filter_by_name_contains(current_indices, filters['name_contains'])
    
    # Website domain filter
    if 'website_domain' in filters:
        current_indices = filter_by_website_domain(current_indices, filters['website_domain'])
    
    return current_indices

# === ADVANCED SEARCH FUNCTIONS ===
def search_companies_with_filters(
    text_query: Optional[str] = None,
    top_k: int = DatabaseConfig.DEFAULT_TOP_K,
    **filters
) -> List[Dict[str, Any]]:
    """
    Advanced search with text query and stackable filters
    
    Args:
        text_query: Optional semantic text search
        top_k: Number of results to return
        **filters: Filter parameters:
            - min_revenue, max_revenue: Revenue range
            - min_team_size, max_team_size: Team size range  
            - min_founded, max_founded: Founded year range
            - industry: Industry name(s) - str or List[str]
            - location: Location(s) - str or List[str]
            - name_contains: Substring in company name
            - website_domain: Domain in website URL
    
    Returns:
        List of company dictionaries sorted by relevance/similarity
    """
    try:
        # If we have a text query, start with semantic search
        if text_query and text_query.strip():
            # Get semantic search results
            semantic_results = search_companies_by_text(text_query, top_k=min(top_k * 3, get_db_length(company_db)))
            candidate_indices = [i for i in range(len(semantic_results))]
            
            # Apply filters to semantic results
            filtered_indices = apply_filters(candidate_indices, filters)
            
            # Return filtered semantic results in original order
            results = []
            for idx in filtered_indices[:top_k]:
                if idx < len(semantic_results):
                    results.append(semantic_results[idx])
            
            return results
        
        else:
            # No text query - just filter all companies
            all_indices = get_all_company_indices()
            filtered_indices = apply_filters(all_indices, filters)
            
            # Convert to company results (no semantic scoring)
            results = []
            for idx in filtered_indices[:top_k]:
                company = get_company(idx, company_db)
                if company:
                    result = {
                        "name": get_name(company),
                        "match_score": None,  # No semantic matching
                        "description": get_description(company),
                        "needs": get_needs(company),
                        "challenges": get_challenges(company),
                        "website": get_website(company),
                        "industry": get_industry(company),
                        "location": get_location(company),
                        "revenue": get_revenue(company),
                        "team_size": get_team_size(company),
                        "founded": get_founded(company)
                    }
                    results.append(result)
            
            return results
            
    except Exception as e:
        raise DatabaseError(f"Failed to search with filters: {str(e)}")

def filter_companies(**filters) -> List[Dict[str, Any]]:
    """
    Pure filtering without text search - returns all companies matching filters
    
    Usage examples:
        filter_companies(industry="FinTech", min_revenue=1000000)
        filter_companies(location=["UK", "USA"], min_team_size=10, max_team_size=50)
        filter_companies(min_founded=2020, industry=["Tech", "AI"])
    """
    return search_companies_with_filters(text_query=None, top_k=get_db_length(company_db), **filters)