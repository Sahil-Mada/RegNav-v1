"""
Document Hierarchy Classifier.
Classifies documents into 7 regulatory hierarchy levels.
"""

from typing import Dict, Optional, List
import re


# 7 Hierarchy Levels for Regulatory Documents
HIERARCHY_LEVELS = [
    "Acts and Laws",
    "Administrative Law / Regulations / Directives / Code of Federal Regulations (CFR)",
    "Pharmacopoeia Standards",
    "Authority Manual / Procedures",
    "Guidances / Guidelines / Q&As",
    "Position Paper",
    "White Papers"
]

# Keywords for each hierarchy level (for automatic classification)
HIERARCHY_KEYWORDS = {
    "Acts and Laws": {
        "keywords": ["act", "law", "statute", "legislation", "congressional", "public law", 
                    "federal food drug and cosmetic act", "fd&c act", "ffdca"],
        "patterns": [r"\b(act|law)\s+\d+", r"public\s+law\s+\d+", r"statute\s+\d+"]
    },
    "Administrative Law / Regulations / Directives / Code of Federal Regulations (CFR)": {
        "keywords": ["cfr", "code of federal regulations", "regulation", "directive", 
                    "administrative law", "21 cfr", "federal register", "final rule"],
        "patterns": [r"\d+\s+cfr", r"21\s+cfr\s+§", r"federal\s+register"]
    },
    "Pharmacopoeia Standards": {
        "keywords": ["pharmacopoeia", "usp", "nf", "monograph", "compendial", 
                    "united states pharmacopeia", "national formulary", "standard"],
        "patterns": [r"\b(usp|nf)\s+\d+", r"monograph", r"compendial"]
    },
    "Authority Manual / Procedures": {
        "keywords": ["manual", "procedure", "sop", "standard operating procedure", 
                    "policy", "authority manual", "operations manual", "compliance manual"],
        "patterns": [r"manual\s+of\s+", r"sop\s+\d+", r"standard\s+operating\s+procedure"]
    },
    "Guidances / Guidelines / Q&As": {
        "keywords": ["guidance", "guideline", "q&a", "questions and answers", 
                    "draft guidance", "final guidance", "recommendation", "best practice"],
        "patterns": [r"guidance\s+(for|on|document)", r"q&a", r"questions?\s+and\s+answers?"]
    },
    "Position Paper": {
        "keywords": ["position paper", "position statement", "statement", "viewpoint", 
                    "perspective", "opinion paper"],
        "patterns": [r"position\s+(paper|statement)", r"viewpoint", r"perspective"]
    },
    "White Papers": {
        "keywords": ["white paper", "whitepaper", "technical paper", "research paper", 
                    "discussion paper", "informational paper"],
        "patterns": [r"white\s+paper", r"whitepaper", r"technical\s+paper"]
    }
}


class DocumentHierarchyClassifier:
    """
    Classifies documents into regulatory hierarchy levels.
    Uses keyword matching and pattern recognition.
    Can be replaced with a trained classifier or LLM-based classification.
    """
    
    def __init__(self, hierarchy_levels: List[str] = None, keywords: Dict = None):
        """
        Initialize the classifier.
        
        Args:
            hierarchy_levels: List of hierarchy level names
            keywords: Dictionary mapping levels to keywords and patterns
        """
        self.hierarchy_levels = hierarchy_levels or HIERARCHY_LEVELS
        self.keywords = keywords or HIERARCHY_KEYWORDS
    
    def classify_document(self, text: str, file_name: str = "") -> Dict[str, any]:
        """
        Classify a document into a hierarchy level.
        
        Args:
            text: Document text
            file_name: Optional file name for additional context
            
        Returns:
            Dictionary with 'level', 'confidence', 'method', and 'scores'
        """
        text_lower = text.lower()
        file_lower = file_name.lower() if file_name else ""
        combined_text = f"{file_lower} {text_lower}"
        
        # Score each hierarchy level
        level_scores = {}
        for level in self.hierarchy_levels:
            config = self.keywords.get(level, {})
            keywords = config.get("keywords", [])
            patterns = config.get("patterns", [])
            
            # Count keyword matches
            keyword_matches = sum(1 for keyword in keywords if keyword.lower() in combined_text)
            
            # Count pattern matches
            pattern_matches = 0
            for pattern in patterns:
                pattern_matches += len(re.findall(pattern, combined_text, re.IGNORECASE))
            
            # Weighted score
            weighted_score = keyword_matches * 2 + pattern_matches * 3
            level_scores[level] = {
                "keyword_matches": keyword_matches,
                "pattern_matches": pattern_matches,
                "weighted": weighted_score
            }
        
        # Find level with highest score
        if not level_scores or max(level_scores.values(), key=lambda x: x["weighted"])["weighted"] == 0:
            # No clear match - return None to indicate user input needed
            return {
                "level": None,
                "confidence": 0.0,
                "method": "keyword_heuristic",
                "scores": level_scores,
                "needs_user_input": True
            }
        
        predicted_level = max(level_scores.items(), key=lambda x: x[1]["weighted"])[0]
        max_score = level_scores[predicted_level]["weighted"]
        total_score = sum(s["weighted"] for s in level_scores.values())
        confidence = min(max_score / max(total_score, 1), 1.0) if total_score > 0 else 0.0
        
        # If confidence is too low, request user input
        needs_user_input = confidence < 0.3
        
        return {
            "level": predicted_level,
            "confidence": confidence,
            "method": "keyword_heuristic",
            "scores": level_scores,
            "needs_user_input": needs_user_input
        }
    
    def get_all_levels(self) -> List[str]:
        """
        Get list of all hierarchy levels.
        
        Returns:
            List of hierarchy level names
        """
        return self.hierarchy_levels

