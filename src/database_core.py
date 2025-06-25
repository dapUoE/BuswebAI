"""
Core database operations for the company database system.
"""

from typing import Dict, List, Any, Optional
from config import DatabaseConfig
from models import ValidationError, DatabaseError, CompanyNotFoundError
from validators import validate_company_data

# === ATOMIC DATABASE FUNCTIONS ===
def append_to_db(data: Dict[str, Any], db_list: List[Dict[str, Any]]) -> None:
    """Append data to database list"""
    if not isinstance(data, dict):
        raise ValidationError("Data must be a dictionary")
    db_list.append(data)

def get_db_length(db_list: List[Any]) -> int:
    """Get length of database list"""
    return len(db_list)

def get_last_index(db_list: List[Any]) -> int:
    """Get the last valid index in the database"""
    length = get_db_length(db_list)
    if length == 0:
        raise DatabaseError("Database is empty")
    return length - 1

def is_index_valid(idx: int) -> bool:
    """Check if index is non-negative"""
    if not isinstance(idx, int):
        return False
    return idx >= 0

def is_index_in_range(idx: int, db_list: List[Any]) -> bool:
    """Check if index is within database range"""
    if not isinstance(idx, int):
        return False
    return 0 <= idx < get_db_length(db_list)

def is_valid_index(idx: int, db_list: List[Any]) -> bool:
    """Check if index is valid and in range"""
    return is_index_valid(idx) and is_index_in_range(idx, db_list)

def get_item_by_index(idx: int, db_list: List[Any]) -> Optional[Any]:
    """Get item by index with validation"""
    if not isinstance(idx, int):
        raise ValidationError("Index must be an integer")
    
    if is_valid_index(idx, db_list):
        return db_list[idx]
    return None

def get_company(idx: int, db_list: List[Any]) -> Optional[Any]:
    """Get company by index - convenience wrapper"""
    return get_item_by_index(idx, db_list)

def copy_db(db_list: List[Any]) -> List[Any]:
    """Get a copy of the database"""
    return db_list.copy()

# === FIELD ACCESS FUNCTIONS ===
def get_field(company: Dict[str, Any], field: str) -> Any:
    """Generic field getter with validation"""
    if not isinstance(company, dict):
        raise ValidationError("Company must be a dictionary")
    if field not in company:
        raise ValidationError(f"Company missing {field} field")
    return company[field]

# Convenience functions for common fields
def get_name(company: Dict[str, Any]) -> str:
    """Get company name"""
    return get_field(company, "name")

def get_industry(company: Dict[str, Any]) -> str:
    """Get company industry"""
    return get_field(company, "industry")

def get_location(company: Dict[str, Any]) -> str:
    """Get company location"""
    return get_field(company, "location")

def get_revenue(company: Dict[str, Any]) -> int:
    """Get company revenue"""
    return get_field(company, "revenue")

def get_team_size(company: Dict[str, Any]) -> int:
    """Get company team size"""
    return get_field(company, "team_size")

def get_founded(company: Dict[str, Any]) -> int:
    """Get company founded year"""
    return get_field(company, "founded")

def get_website(company: Dict[str, Any]) -> str:
    """Get company website"""
    return get_field(company, "website")

def get_description(company: Dict[str, Any]) -> str:
    """Get company description"""
    return get_field(company, "description")

def get_needs(company: Dict[str, Any]) -> str:
    """Get company needs"""
    return get_field(company, "needs")

def get_challenges(company: Dict[str, Any]) -> str:
    """Get company challenges"""
    return get_field(company, "challenges")

def create_company_dict(name: str, industry: str, location: str, revenue: int, 
                       team_size: int, founded: int, website: str, 
                       description: str, needs: str, challenges: str) -> Dict[str, Any]:
    """Create and validate company dictionary"""
    return validate_company_data(name, industry, location, revenue, team_size, 
                                founded, website, description, needs, challenges)