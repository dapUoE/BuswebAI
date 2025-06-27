"""
Configuration and constants for the company database system.
"""

from typing import List

class DatabaseConfig:
    """Configuration constants for the database system"""
    
    # Model Configuration
    OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small'
    EMBEDDING_DIMENSION = 1536  # text-embedding-3-small dimension
    
    # Search Configuration
    DEFAULT_TOP_K = 3
    DEFAULT_SCORE_DECIMALS = 3
    
    # Field Names
    REQUIRED_FIELDS = [
        'name', 'industry', 'location', 'revenue', 'team_size', 
        'founded', 'website', 'description', 'needs', 'challenges'
    ]
    
    # Validation Constants
    MIN_REVENUE = 0
    MAX_REVENUE = 1_000_000_000_000  # 1 trillion
    MIN_TEAM_SIZE = 1
    MAX_TEAM_SIZE = 1_000_000
    MIN_FOUNDED_YEAR = 1800
    MAX_FOUNDED_YEAR = 2100
    MAX_STRING_LENGTH = 10000