"""
Input validation functions for the company database system.
"""

from typing import Any, Dict
from config import DatabaseConfig
from models import ValidationError

def validate_string_field(value: Any, field_name: str, required: bool = True) -> str:
    """Validate string field with length and requirement checks"""
    if value is None:
        if required:
            raise ValidationError(f"{field_name} is required")
        return ""
    
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    
    value = value.strip()
    if required and not value:
        raise ValidationError(f"{field_name} cannot be empty")
    
    if len(value) > DatabaseConfig.MAX_STRING_LENGTH:
        raise ValidationError(f"{field_name} too long (max {DatabaseConfig.MAX_STRING_LENGTH} characters)")
    
    return value

def validate_integer_field(value: Any, field_name: str, min_val: int, max_val: int) -> int:
    """Validate integer field with range checks"""
    if value is None:
        raise ValidationError(f"{field_name} is required")
    
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a number")
    
    value = int(value)
    if value < min_val or value > max_val:
        raise ValidationError(f"{field_name} must be between {min_val} and {max_val}")
    
    return value

def validate_company_data(name: Any, industry: Any, location: Any, revenue: Any, 
                         team_size: Any, founded: Any, website: Any, 
                         description: Any, needs: Any, challenges: Any) -> Dict[str, Any]:
    """Validate all company fields and return clean data"""
    try:
        validated_data = {
            'name': validate_string_field(name, 'name'),
            'industry': validate_string_field(industry, 'industry'),
            'location': validate_string_field(location, 'location'),
            'revenue': validate_integer_field(revenue, 'revenue', DatabaseConfig.MIN_REVENUE, DatabaseConfig.MAX_REVENUE),
            'team_size': validate_integer_field(team_size, 'team_size', DatabaseConfig.MIN_TEAM_SIZE, DatabaseConfig.MAX_TEAM_SIZE),
            'founded': validate_integer_field(founded, 'founded', DatabaseConfig.MIN_FOUNDED_YEAR, DatabaseConfig.MAX_FOUNDED_YEAR),
            'website': validate_string_field(website, 'website'),
            'description': validate_string_field(description, 'description'),
            'needs': validate_string_field(needs, 'needs'),
            'challenges': validate_string_field(challenges, 'challenges')
        }
        return validated_data
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f"Validation failed: {str(e)}")