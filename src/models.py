"""
Data models and exception classes for the company database system.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any

# === EXCEPTION CLASSES ===
class DatabaseError(Exception):
    """Base exception for database operations"""
    pass

class ValidationError(DatabaseError):
    """Raised when input validation fails"""
    pass

class CompanyNotFoundError(DatabaseError):
    """Raised when a company is not found"""
    pass

class EmbeddingError(DatabaseError):
    """Raised when embedding operations fail"""
    pass

# === COMPANY DATA STRUCTURE ===
@dataclass
class Company:
    """Company data structure with type hints and validation"""
    name: str
    industry: str
    location: str
    revenue: int
    team_size: int
    founded: int
    website: str
    description: str
    needs: str
    challenges: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Company to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Company':
        """Create Company from dictionary"""
        return cls(**data)