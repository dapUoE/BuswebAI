"""
Tag generation service using OpenAI ChatGPT API.
Converts company descriptions into structured tags for better comparison.
"""

import os
import json
from typing import List, Dict, Any
from openai import OpenAI
from models import ValidationError, EmbeddingError


class TagGenerator:
    """Service for generating structured tags from company descriptions using ChatGPT."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize the tag generator with OpenAI client."""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model
        
        # Define tag categories and examples
        self.tag_categories = {
            "industry": [
                "fintech", "healthcare", "ai-ml", "blockchain", "cybersecurity",
                "edtech", "proptech", "insurtech", "biotech", "cleantech",
                "retail", "manufacturing", "logistics", "automotive", "agriculture"
            ],
            "technology": [
                "machine-learning", "artificial-intelligence", "web-development",
                "mobile-apps", "cloud-computing", "data-analytics", "iot",
                "robotics", "ar-vr", "api-development", "devops", "blockchain-tech"
            ],
            "business_model": [
                "b2b", "b2c", "b2b2c", "saas", "marketplace", "subscription",
                "freemium", "enterprise", "consulting", "licensing", "advertising"
            ],
            "stage": [
                "pre-seed", "seed", "series-a", "series-b", "series-c",
                "growth", "mature", "public", "startup", "scale-up"
            ],
            "market": [
                "enterprise", "consumer", "smb", "government", "healthcare-providers",
                "financial-institutions", "education", "retail-chains", "startups"
            ],
            "solution_type": [
                "platform", "tool", "service", "infrastructure", "analytics",
                "automation", "integration", "security", "compliance", "optimization"
            ]
        }
    
    def generate_tags(self, description: str) -> Dict[str, List[str]]:
        """
        Generate structured tags from a company description.
        
        Args:
            description: Company description text
            
        Returns:
            Dictionary with categorized tags
            
        Raises:
            ValidationError: If description is invalid
            EmbeddingError: If API call fails
        """
        if not isinstance(description, str):
            raise ValidationError("Description must be a string")
        
        if not description.strip():
            raise ValidationError("Description cannot be empty")
        
        try:
            # Create the prompt for structured tag extraction
            prompt = self._create_tag_extraction_prompt(description)
            
            # Call ChatGPT API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert business analyst who extracts structured tags from company descriptions. Always respond with valid JSON."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500
            )
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            
            # Clean up the response if it has markdown formatting
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            tags = json.loads(content)
            
            # Validate the response structure
            validated_tags = self._validate_and_clean_tags(tags)
            
            return validated_tags
            
        except json.JSONDecodeError as e:
            raise EmbeddingError(f"Failed to parse tag response as JSON: {str(e)}")
        except Exception as e:
            raise EmbeddingError(f"Failed to generate tags: {str(e)}")
    
    def _create_tag_extraction_prompt(self, description: str) -> str:
        """Create a prompt for tag extraction."""
        
        # Create examples for each category
        category_examples = {}
        for category, tags in self.tag_categories.items():
            category_examples[category] = tags[:8]  # Show first 8 examples
        
        prompt = f"""
Extract structured tags from the following company description. Return a JSON object with tags categorized as follows:

CATEGORIES AND EXAMPLES:
{json.dumps(category_examples, indent=2)}

RULES:
1. Select 2-5 most relevant tags per category
2. Only use tags that clearly apply to the company
3. Prefer specific over general tags
4. If a category doesn't apply, use an empty array []
5. You can create new tags if none of the examples fit, but keep them in the same style
6. Return valid JSON only

COMPANY DESCRIPTION:
{description}

Return JSON in this exact format:
{{
  "industry": ["tag1", "tag2"],
  "technology": ["tag1", "tag2"],
  "business_model": ["tag1"],
  "stage": ["tag1"],
  "market": ["tag1", "tag2"],
  "solution_type": ["tag1"]
}}
"""
        return prompt
    
    def _validate_and_clean_tags(self, tags: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate and clean the generated tags."""
        validated = {}
        
        for category in self.tag_categories.keys():
            if category in tags:
                category_tags = tags[category]
                if isinstance(category_tags, list):
                    # Clean and validate each tag
                    cleaned_tags = []
                    for tag in category_tags:
                        if isinstance(tag, str) and tag.strip():
                            # Normalize tag format
                            cleaned_tag = tag.strip().lower().replace(" ", "-")
                            cleaned_tags.append(cleaned_tag)
                    validated[category] = cleaned_tags[:5]  # Max 5 tags per category
                else:
                    validated[category] = []
            else:
                validated[category] = []
        
        return validated
    
    def tags_to_string(self, tags: Dict[str, List[str]]) -> str:
        """Convert tags dictionary to a string for embedding."""
        tag_strings = []
        
        for category, category_tags in tags.items():
            if category_tags:
                # Create category-prefixed strings
                prefixed_tags = [f"{category}:{tag}" for tag in category_tags]
                tag_strings.extend(prefixed_tags)
        
        return " ".join(tag_strings)
    
    def get_all_tags_flat(self, tags: Dict[str, List[str]]) -> List[str]:
        """Get all tags as a flat list."""
        all_tags = []
        for category_tags in tags.values():
            all_tags.extend(category_tags)
        return all_tags


# Convenience function for quick tag generation
def generate_company_tags(description: str) -> Dict[str, List[str]]:
    """Generate tags for a company description."""
    generator = TagGenerator()
    return generator.generate_tags(description)


# Example usage and testing
if __name__ == "__main__":
    # Test the tag generator
    test_descriptions = [
        "AI-powered healthcare platform that uses machine learning to analyze medical images and assist radiologists in early disease detection",
        "B2B SaaS fintech company providing automated accounting and financial management tools for small businesses",
        "Consumer mobile app marketplace connecting freelance graphic designers with small business owners"
    ]
    
    generator = TagGenerator()
    
    for i, desc in enumerate(test_descriptions, 1):
        print(f"\n=== Test {i} ===")
        print(f"Description: {desc}")
        
        try:
            tags = generator.generate_tags(desc)
            print(f"Generated tags: {json.dumps(tags, indent=2)}")
            
            # Show string representation
            tag_string = generator.tags_to_string(tags)
            print(f"Tag string: {tag_string}")
            
        except Exception as e:
            print(f"Error: {e}")