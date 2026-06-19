"""
Web Fallback Module - Bit 4 (WORKING VERSION)
Uses ddgs (new DuckDuckGo package) for reliable web search
"""

import re
import time
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Try to import the new ddgs package
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
    print("✅ DDGS (new package) available")
except ImportError:
    try:
        from duckduckgo_search import DDGS
        DDGS_AVAILABLE = True
        print("✅ DDGS (old package) available")
    except ImportError:
        DDGS_AVAILABLE = False
        print("❌ DDGS not available - install with: pip install ddgs")

class WebFallback:
    """
    Web search fallback using DDGS (DuckDuckGo Search)
    """
    
    def __init__(self, max_results: int = 5):
        """
        Initialize web fallback
        
        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
        print(f"✅ Web Fallback initialized (max_results={max_results})")
    
    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """
        Perform web search using DDGS
        
        Args:
            query: Search query
            max_results: Max results (overrides default)
            
        Returns:
            List of search results with title, body, URL, and metadata
        """
        if max_results is None:
            max_results = self.max_results
        
        results = []
        
        # Try DDGS search
        if DDGS_AVAILABLE:
            results = self._search_ddgs(query, max_results)
            if results:
                return results
        
        # Fallback: Try wttr.in for weather queries
        if not results and ('weather' in query.lower() or 'temperature' in query.lower()):
            weather_results = self._get_weather_data(query)
            if weather_results:
                return weather_results
        
        # Fallback: Try Wikipedia for general knowledge
        if not results:
            wiki_results = self._search_wikipedia(query, max_results)
            if wiki_results:
                return wiki_results
        
        return results
    
    def _search_ddgs(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DDGS (DuckDuckGo Search)"""
        try:
            print(f"🔍 Searching web for: {query}")
            
            with DDGS() as ddgs:
                # Use text search
                results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    safesearch='moderate',
                    timelimit='m'  # Past month for fresh results
                ))
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                # Get body text
                body = result.get('body', '')
                if not body:
                    body = result.get('description', '')
                if not body:
                    body = result.get('snippet', '')
                
                # Get URL
                url = result.get('href', '')
                if not url:
                    url = result.get('url', '')
                
                formatted_results.append({
                    'id': i,
                    'title': result.get('title', 'No Title'),
                    'body': body,
                    'url': url,
                    'source': url,
                    'relevance_score': self._calculate_relevance(body, query)
                })
            
            if formatted_results:
                print(f"✅ DDGS found {len(formatted_results)} results")
                # Print first result preview
                print(f"   First result: {formatted_results[0]['title'][:50]}...")
            else:
                print("⚠️ DDGS returned 0 results")
            
            return formatted_results
            
        except Exception as e:
            print(f"⚠️ DDGS search failed: {e}")
            return []
    
    def _get_weather_data(self, query: str) -> List[Dict[str, Any]]:
        """Get weather data using wttr.in API (free, no key needed)"""
        try:
            # Extract location from query
            location = query.lower()
            # Remove weather-related words
            for word in ['weather', 'temperature', 'in', 'what', 'is', 'the', 'current', 'today', 'now', 'like']:
                location = location.replace(word, '')
            location = location.strip()
            
            if not location:
                return []
            
            print(f"   🌤️ Getting weather for: {location}")
            
            # Use wttr.in API
            url = f"https://wttr.in/{location}?format=%C+%t+%w&lang=en"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                weather_text = response.text.strip()
                if weather_text and 'Unknown' not in weather_text:
                    # Parse weather data
                    parts = weather_text.split()
                    condition = ' '.join(parts[:-2]) if len(parts) > 2 else 'Unknown'
                    temp = parts[-2] if len(parts) >= 2 else 'Unknown'
                    wind = parts[-1] if len(parts) >= 1 else 'Unknown'
                    
                    return [{
                        'id': 1,
                        'title': f"Weather in {location.title()}",
                        'body': f"Current weather: {condition}, Temperature: {temp}, Wind: {wind}",
                        'url': f"https://wttr.in/{location}",
                        'source': f"https://wttr.in/{location}",
                        'relevance_score': 0.95
                    }]
            
            return []
            
        except Exception as e:
            print(f"⚠️ Weather API failed: {e}")
            return []
    
    def _search_wikipedia(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Wikipedia as a fallback"""
        try:
            print(f"🔍 Searching Wikipedia for: {query}")
            
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": max_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('query', {}).get('search', [])
                
                formatted_results = []
                for i, result in enumerate(results, 1):
                    formatted_results.append({
                        'id': i,
                        'title': result.get('title', 'No Title'),
                        'body': result.get('snippet', 'No Content').replace('<span class="searchmatch">', '').replace('</span>', ''),
                        'url': f"https://en.wikipedia.org/wiki/{result.get('title', '').replace(' ', '_')}",
                        'source': 'Wikipedia',
                        'relevance_score': 0.7
                    })
                
                if formatted_results:
                    print(f"✅ Wikipedia found {len(formatted_results)} results")
                return formatted_results
            
            return []
            
        except Exception as e:
            print(f"⚠️ Wikipedia search failed: {e}")
            return []
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score for a search result"""
        if not content:
            return 0.0
        
        content_lower = content.lower()
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Count matching words
        matches = 0
        for word in query_words:
            if len(word) > 2 and word in content_lower:
                matches += 1
        
        # Calculate score based on match ratio
        match_ratio = matches / len(query_words) if query_words else 0
        score = min(match_ratio * 1.5, 1.0)
        
        if query_lower in content_lower:
            score = min(score + 0.3, 1.0)
        
        return round(score, 3)
    
    def search_with_sources(self, query: str, max_results: int = None) -> Dict[str, Any]:
        """Search and format results with source tracking"""
        results = self.search(query, max_results)
        
        if not results:
            return {
                'results': [],
                'context': "",
                'sources': [],
                'num_results': 0,
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
        
        context_parts = []
        sources = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Web Source {i}]: {result['body']}")
            
            sources.append({
                'id': result['id'],
                'title': result['title'],
                'content': result['body'],
                'url': result['url'],
                'source_type': 'web',
                'relevance': result['relevance_score'],
                'source_display': f"🔗 {result['title']}" if result['title'] else f"🔗 Web Search"
            })
        
        return {
            'results': results,
            'context': "\n\n".join(context_parts),
            'sources': sources,
            'num_results': len(results),
            'query': query,
            'timestamp': datetime.now().isoformat()
        }
    
    def smart_search(self, query: str, router_decision: Dict[str, Any] = None) -> Dict[str, Any]:
        """Smart search that adapts based on router decision"""
        if router_decision:
            confidence = router_decision.get('confidence', 0.5)
            if confidence > 0.7:
                max_results = self.max_results
            elif confidence > 0.4:
                max_results = max(2, self.max_results // 2)
            else:
                max_results = 2
        else:
            max_results = self.max_results
        
        search_results = self.search_with_sources(query, max_results)
        
        if search_results['num_results'] == 0:
            search_results['recommendation'] = "No web results found. Try refining your query."
        elif search_results['num_results'] < 3:
            search_results['recommendation'] = "Limited web results. Consider using documents if available."
        else:
            search_results['recommendation'] = f"Found {search_results['num_results']} web results."
        
        return search_results


# Test function
def test_web_fallback():
    """Test the web fallback module"""
    print("=" * 60)
    print("🧪 BIT 4: Testing Web Fallback")
    print("=" * 60)
    
    # Initialize web fallback
    web_fallback = WebFallback(max_results=3)
    
    # Test queries
    test_queries = [
        "weather in Lagos Nigeria",
        "current temperature in London",
        "Who is the president of Nigeria",
        "What is Python programming language"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        try:
            results = web_fallback.search_with_sources(query, max_results=3)
            
            print(f"🔍 Found: {results['num_results']} results")
            
            if results['num_results'] > 0:
                for i, source in enumerate(results['sources'][:2], 1):
                    print(f"\nResult {i}:")
                    print(f"  Title: {source['title'][:60]}...")
                    print(f"  Content: {source['content'][:100]}...")
                    print(f"  Relevance: {source['relevance']}")
                    print(f"  URL: {source['url'][:50]}...")
            else:
                print("  No results found")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Web Fallback Module Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_web_fallback()