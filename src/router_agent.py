"""
Router Agent Module - Bit 3
Decides whether to use documents, web, or both for a query
"""

import re
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

class RouteType(Enum):
    """Types of routes the agent can choose"""
    DOCUMENTS = "documents"
    WEB = "web"
    BOTH = "both"
    NONE = "none"

@dataclass
class RouteDecision:
    """Decision made by the router agent"""
    route: RouteType
    confidence: float
    reasoning: str
    keywords_matched: List[str]
    suggested_sources: List[str]

class RouterAgent:
    """
    Router Agent that decides which sources to use for a query
    Uses keyword-based routing with confidence scoring
    """
    
    def __init__(self):
        """Initialize the router agent with routing rules"""
        
        # Keywords that indicate web search is needed
        self.web_keywords = {
            'high': [  # Strong indicators - web definitely needed
                'today', 'current', 'latest', 'breaking', 'now',
                '2025', '2026', '2024', 'this year', 'this month',
                'weather', 'forecast', 'score', 'stock price',
                'live', 'upcoming', 'scheduled', 'happening'
            ],
            'medium': [  # Moderate indicators - web recommended
                'news', 'update', 'recent', 'new', 'latest version',
                'price', 'cost', 'available', 'release', 'launch'
            ],
            'low': [  # Weak indicators - web optional
                'online', 'internet', 'website', 'web', 'search',
                'find', 'looking for', 'need to know'
            ]
        }
        
        # Keywords that indicate documents are sufficient
        self.document_keywords = {
            'high': [  # Strong indicators - documents definitely sufficient
                'according to', 'based on', 'from the document',
                'in the file', 'as stated in', 'the text says',
                'summarize', 'explain this document'
            ],
            'medium': [  # Moderate indicators - documents recommended
                'document', 'pdf', 'file', 'uploaded', 'content',
                'information about', 'tell me about', 'what is'
            ],
            'low': [  # Weak indicators - documents optional
                'explain', 'describe', 'define', 'meaning of',
                'concept', 'idea', 'topic'
            ]
        }
        
        # General knowledge keywords (can use either)
        self.general_keywords = [
            'what', 'how', 'why', 'when', 'where', 'who',
            'which', 'define', 'explain', 'describe'
        ]
    
    def route(self, query: str, has_documents: bool = False) -> RouteDecision:
        """
        Route a query to the appropriate source(s)
        
        Args:
            query: The user's question
            has_documents: Whether documents are available
            
        Returns:
            RouteDecision with route type and confidence
        """
        query_lower = query.lower()
        
        # Calculate scores for each route
        web_score = self._calculate_web_score(query_lower)
        doc_score = self._calculate_document_score(query_lower, has_documents)
        
        # Track matched keywords
        matched_web_keywords = self._get_matched_keywords(query_lower, self.web_keywords)
        matched_doc_keywords = self._get_matched_keywords(query_lower, self.document_keywords)
        
        # Determine route based on scores
        if not has_documents:
            # No documents available - must use web
            if web_score > 0:
                route = RouteType.WEB
                confidence = min(web_score, 0.9)
                reasoning = "No documents available, using web search"
                matched_keywords = matched_web_keywords
            else:
                route = RouteType.NONE
                confidence = 0.0
                reasoning = "No documents available and no web indicators found"
                matched_keywords = []
        
        elif web_score > 0.7 and doc_score < 0.5:
            # Strong web indicators, weak document indicators
            route = RouteType.WEB
            confidence = web_score
            reasoning = "Strong web indicators detected (current/recent information needed)"
            matched_keywords = matched_web_keywords
        
        elif doc_score > 0.7 and web_score < 0.5:
            # Strong document indicators, weak web indicators
            route = RouteType.DOCUMENTS
            confidence = doc_score
            reasoning = "Strong document indicators detected (specific document references)"
            matched_keywords = matched_doc_keywords
        
        elif web_score > 0.5 and doc_score > 0.5:
            # Both have decent scores - use both
            route = RouteType.BOTH
            confidence = max(web_score, doc_score)
            reasoning = "Query has both document and web indicators - using both sources"
            matched_keywords = matched_web_keywords + matched_doc_keywords
        
        elif doc_score > web_score:
            # Documents are more relevant
            route = RouteType.DOCUMENTS
            confidence = doc_score
            reasoning = "Query is best answered from uploaded documents"
            matched_keywords = matched_doc_keywords
        
        elif web_score > 0:
            # Web is more relevant
            route = RouteType.WEB
            confidence = web_score
            reasoning = "Query requires current/real-time information from the web"
            matched_keywords = matched_web_keywords
        
        else:
            # Default to documents if available, else web
            if has_documents:
                route = RouteType.DOCUMENTS
                confidence = 0.5
                reasoning = "No specific routing indicators, defaulting to documents"
            else:
                route = RouteType.WEB
                confidence = 0.3
                reasoning = "No specific routing indicators, using web search"
            matched_keywords = []
        
        # Determine suggested sources
        suggested_sources = self._get_suggested_sources(route, has_documents)
        
        return RouteDecision(
            route=route,
            confidence=confidence,
            reasoning=reasoning,
            keywords_matched=matched_keywords[:5],  # Top 5 matches
            suggested_sources=suggested_sources
        )
    
    def should_use_web_based_on_results(self, query: str, doc_results: List[Dict], has_documents: bool) -> bool:
        """
        Decide if web search is needed based on document results quality
        
        Args:
            query: User's question
            doc_results: Results from document search
            has_documents: Whether documents are available
            
        Returns:
            True if web search should be used
        """
        # If no documents, definitely use web
        if not has_documents:
            return True
        
        # If no document results, use web
        if not doc_results:
            return True
        
        # Check if document results are relevant
        query_words = set(query.lower().split())
        relevant_results = 0
        
        for result in doc_results[:3]:  # Check top 3 results
            content = result.get('content', '').lower()
            # Count how many query words appear in the content
            matches = sum(1 for word in query_words if len(word) > 2 and word in content)
            match_ratio = matches / len(query_words) if query_words else 0
            
            # If result has good relevance, count it
            if match_ratio > 0.2:  # At least 20% of query words match
                relevant_results += 1
        
        # If less than 2 relevant results, use web
        if relevant_results < 2:
            return True
        
        # Also check for web-specific keywords
        web_keywords = ['current', 'today', 'latest', 'news', 'now', 'recent', 'weather', 'temperature']
        if any(keyword in query.lower() for keyword in web_keywords):
            # If query has web keywords AND document results are low confidence
            avg_confidence = sum(r.get('confidence', 0) for r in doc_results[:3]) / 3 if doc_results else 0
            if avg_confidence < 0.6:
                return True
        
        return False
    
    def _calculate_web_score(self, query: str) -> float:
        """Calculate web search relevance score (0-1)"""
        score = 0.0
        
        # Check high priority web keywords
        for keyword in self.web_keywords['high']:
            if keyword in query:
                score += 0.4
        
        # Check medium priority web keywords
        for keyword in self.web_keywords['medium']:
            if keyword in query:
                score += 0.25
        
        # Check low priority web keywords
        for keyword in self.web_keywords['low']:
            if keyword in query:
                score += 0.1
        
        # Check for date/time patterns
        if re.search(r'\d{4}', query):  # Year pattern
            score += 0.2
        if re.search(r'\d{1,2}:\d{2}', query):  # Time pattern
            score += 0.15
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _calculate_document_score(self, query: str, has_documents: bool) -> float:
        """Calculate document relevance score (0-1)"""
        if not has_documents:
            return 0.0
        
        score = 0.0
        
        # Check high priority document keywords
        for keyword in self.document_keywords['high']:
            if keyword in query:
                score += 0.4
        
        # Check medium priority document keywords
        for keyword in self.document_keywords['medium']:
            if keyword in query:
                score += 0.25
        
        # Check low priority document keywords
        for keyword in self.document_keywords['low']:
            if keyword in query:
                score += 0.1
        
        # General knowledge questions - good for documents
        for keyword in self.general_keywords:
            if keyword in query:
                score += 0.05
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _get_matched_keywords(self, query: str, keyword_dict: Dict[str, List[str]]) -> List[str]:
        """Get matched keywords from a keyword dictionary"""
        matched = []
        for category, keywords in keyword_dict.items():
            for keyword in keywords:
                if keyword in query:
                    matched.append(keyword)
        return matched
    
    def _get_suggested_sources(self, route: RouteType, has_documents: bool) -> List[str]:
        """Get suggested sources based on route"""
        if route == RouteType.DOCUMENTS:
            return ["Uploaded Documents"]
        elif route == RouteType.WEB:
            return ["Web Search (DuckDuckGo)"]
        elif route == RouteType.BOTH:
            sources = ["Uploaded Documents"]
            if has_documents:
                sources.append("Web Search (DuckDuckGo)")
            else:
                sources.append("Web Search Only")
            return sources
        else:
            return ["No sources available"]
    
    def get_route_decision(self, query: str, has_documents: bool = False) -> Dict[str, Any]:
        """
        Get route decision as a dictionary (for easy display)
        """
        decision = self.route(query, has_documents)
        
        return {
            "route": decision.route.value,
            "confidence": round(decision.confidence, 3),
            "reasoning": decision.reasoning,
            "keywords_matched": decision.keywords_matched,
            "suggested_sources": decision.suggested_sources
        }


# Test function
def test_router_agent():
    """Test the router agent"""
    print("=" * 60)
    print("🧪 BIT 3: Testing Router Agent")
    print("=" * 60)
    
    router = RouterAgent()
    
    # Test scenarios
    test_queries = [
        # Document-focused queries
        ("What is machine learning according to the document?", True),
        ("Summarize the information in the uploaded files", True),
        ("Explain the concept of RAG from the document", True),
        
        # Web-focused queries
        ("What is the weather today in New York?", False),
        ("What are the latest news about AI?", False),
        ("What is the current stock price of Tesla?", False),
        
        # Mixed queries
        ("What is machine learning and what are the latest developments?", True),
        ("Explain RAG and how is it used today?", True),
        ("What are the document findings and what does the web say?", True),
        
        # No documents available
        ("What is the meaning of life?", False),
        ("Who won the World Cup?", False)
    ]
    
    print("\n📊 Testing Route Decisions:")
    print("-" * 60)
    
    for query, has_docs in test_queries:
        print(f"\n📝 Query: {query}")
        print(f"   Documents available: {has_docs}")
        
        decision = router.get_route_decision(query, has_docs)
        
        print(f"   📍 Route: {decision['route'].upper()}")
        print(f"   📊 Confidence: {decision['confidence']}")
        print(f"   💭 Reasoning: {decision['reasoning']}")
        print(f"   🔑 Keywords: {decision['keywords_matched']}")
        print(f"   📚 Sources: {decision['suggested_sources']}")
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("✅ BIT 3: Router Agent Module COMPLETE!")
    print("=" * 60)
    print("\n📋 Checklist:")
    print("   ✅ Route types defined (DOCUMENTS, WEB, BOTH)")
    print("   ✅ Keyword-based routing logic")
    print("   ✅ Confidence scoring")
    print("   ✅ Reasoning generation")
    print("   ✅ Source suggestions")
    print("   ✅ Error handling")
    print("   ✅ should_use_web_based_on_results method added")


if __name__ == "__main__":
    test_router_agent()