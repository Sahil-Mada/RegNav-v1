"""
Domain Router for Hierarchical RAG.
Routes user queries to the appropriate domain/section.
Currently uses a keyword-based heuristic (placeholder).
Can be replaced with a fine-tuned classifier or zero-shot LLM.
"""

from typing import List, Dict, Optional
import re


# Predefined domain hierarchy (placeholders - can be modified)
DOMAIN_HIERARCHY = {
    "Clinical": {
        "keywords": ["clinical", "trial", "patient", "safety", "efficacy", "adverse", "dosing", 
                    "pharmacokinetics", "pharmacodynamics", "endpoint", "protocol", "informed consent"],
        "subdomains": ["Phase I", "Phase II", "Phase III", "Safety", "Efficacy", "PK/PD"]
    },
    "Nonclinical": {
        "keywords": ["nonclinical", "preclinical", "toxicology", "animal", "in vivo", "in vitro",
                    "carcinogenicity", "genotoxicity", "reproductive", "pharmacology"],
        "subdomains": ["Toxicology", "Pharmacology", "ADME", "Safety Pharmacology"]
    },
    "CMC": {
        "keywords": ["cmc", "chemistry", "manufacturing", "controls", "specification", "analytical",
                    "process", "validation", "stability", "impurity", "formulation"],
        "subdomains": ["Drug Substance", "Drug Product", "Analytical Methods", "Manufacturing"]
    },
    "Compliance": {
        "keywords": ["compliance", "regulation", "inspection", "violation", "warning letter",
                    "483", "capa", "corrective action", "audit"],
        "subdomains": ["Inspections", "Warning Letters", "CAPA", "Audits"]
    },
    "Quality Systems": {
        "keywords": ["quality", "qms", "quality system", "gmp", "good manufacturing practice",
                    "documentation", "change control", "validation", "qualification"],
        "subdomains": ["QMS", "GMP", "Documentation", "Change Control"]
    },
    "Submission Logistics": {
        "keywords": ["submission", "nda", "ind", "nda", "bla", "filing", "review", "meeting",
                    "pdufa", "user fee", "timeline", "milestone"],
        "subdomains": ["IND", "NDA", "BLA", "Meetings", "Timelines"]
    },
    "Labeling": {
        "keywords": ["labeling", "label", "package insert", "prescribing information", "indication",
                    "contraindication", "warning", "precaution", "adverse reaction"],
        "subdomains": ["Package Insert", "Indications", "Warnings", "Dosage"]
    }
}


class DomainRouter:
    """
    Routes queries to appropriate domains using keyword-based heuristics.
    This is a placeholder implementation - can be replaced with:
    - Fine-tuned classification model
    - Zero-shot LLM classifier
    - More sophisticated NLP-based classifier
    """
    
    def __init__(self, domain_hierarchy: Dict = None):
        """
        Initialize the router with domain hierarchy.
        
        Args:
            domain_hierarchy: Dictionary mapping domains to keywords and subdomains
        """
        self.domain_hierarchy = domain_hierarchy or DOMAIN_HIERARCHY
        self.domains = list(self.domain_hierarchy.keys())
    
    def route_query(self, query: str) -> Dict[str, any]:
        """
        Route a user query to the most appropriate domain.
        
        Args:
            query: User's question/query
            
        Returns:
            Dictionary with 'domain', 'confidence', and 'method'
        """
        query_lower = query.lower()
        
        # Score each domain based on keyword matches
        domain_scores = {}
        for domain, config in self.domain_hierarchy.items():
            keywords = config.get("keywords", [])
            score = sum(1 for keyword in keywords if keyword.lower() in query_lower)
            # Weight by keyword length (longer keywords are more specific)
            weighted_score = sum(len(keyword) for keyword in keywords if keyword.lower() in query_lower)
            domain_scores[domain] = {
                "count": score,
                "weighted": weighted_score
            }
        
        # Find domain with highest score
        if not domain_scores or max(domain_scores.values(), key=lambda x: x["weighted"])["weighted"] == 0:
            # Default to first domain if no matches
            predicted_domain = self.domains[0]
            confidence = 0.1
        else:
            predicted_domain = max(domain_scores.items(), key=lambda x: x[1]["weighted"])[0]
            max_score = domain_scores[predicted_domain]["weighted"]
            total_score = sum(s["weighted"] for s in domain_scores.values())
            confidence = min(max_score / max(total_score, 1), 1.0) if total_score > 0 else 0.1
        
        return {
            "domain": predicted_domain,
            "confidence": confidence,
            "method": "keyword_heuristic",
            "scores": domain_scores
        }
    
    def get_subdomains(self, domain: str) -> List[str]:
        """
        Get list of subdomains for a given domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of subdomain names
        """
        return self.domain_hierarchy.get(domain, {}).get("subdomains", [])
    
    def get_all_domains(self) -> List[str]:
        """
        Get list of all available domains.
        
        Args:
            None
            
        Returns:
            List of all domain names
        """
        return self.domains

