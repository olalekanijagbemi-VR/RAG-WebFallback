"""
Answer Generation Module - Bit 5
Generates answers using Groq with proper citation and source tracking
"""

import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

from groq import Groq

# Don't import streamlit here - it causes issues in testing
# We'll only use it when needed

class AnswerGenerator:
    """
    Answer generator using Groq's Llama 3.1
    Generates answers with source citations
    """
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize answer generator
        
        Args:
            model_name: Groq model to use
        """
        self.model_name = model_name
        
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('GROQ_API_KEY')
        
        print(f"🔑 API Key loaded: {'✅ Yes' if self.api_key else '❌ No'}")
        if self.api_key:
            print(f"   API Key starts with: {self.api_key[:10]}...")
        
        if not self.api_key:
            print("❌ GROQ_API_KEY not found in environment variables")
            print("   Please make sure .env file exists with GROQ_API_KEY=your_key")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
                print(f"✅ Groq client initialized with model: {model_name}")
            except Exception as e:
                print(f"❌ Error initializing Groq: {e}")
                self.client = None
    
    def generate_answer(self, query: str, context: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Generate an answer with source citations
        
        Args:
            query: User's question
            context: Retrieved context from documents or web
            sources: List of sources with metadata
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.client:
            return {
                'answer': "❌ Error: Groq client not initialized. Please check your API key.",
                'sources': sources,
                'error': True
            }
        
        if not context:
            return {
                'answer': "I couldn't find relevant information to answer your question. Please try rephrasing or upload more documents.",
                'sources': [],
                'error': False
            }
        
        try:
            # Build prompt with source citations
            prompt = self._build_prompt(query, context, sources)
            
            # Get response from Groq
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Parse citations from answer
            cited_sources = self._extract_citations(answer, sources)
            
            return {
                'answer': answer,
                'sources': sources,
                'cited_sources': cited_sources,
                'model': self.model_name,
                'timestamp': datetime.now().isoformat(),
                'error': False
            }
            
        except Exception as e:
            return {
                'answer': f"❌ Error generating answer: {str(e)}",
                'sources': sources,
                'error': True
            }
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the LLM"""
        return """You are a helpful assistant that answers questions based on provided context.

        Instructions:
        1. Answer ONLY based on the provided context
        2. Cite sources using [Source X] format where X is the source number
        3. If the context doesn't contain the answer, say "I couldn't find information about this in the provided sources"
        4. Be concise but thorough
        5. Organize information clearly
        6. If multiple sources are relevant, mention all of them"""
    
    def _build_prompt(self, query: str, context: str, sources: List[Dict]) -> str:
        """Build the prompt with context and source references"""
        
        # Format sources for the prompt
        source_list = []
        for i, source in enumerate(sources, 1):
            source_type = source.get('source_type', 'unknown')
            source_name = source.get('source', source.get('title', 'Unknown'))
            content = source.get('content', '')
            
            source_list.append(f"[Source {i}] ({source_type} from {source_name}):\n{content}")
        
        sources_text = "\n\n".join(source_list)
        
        prompt = f"""Context Information:
        {sources_text}

        Question: {query}

        Please provide a clear, concise answer based on the context above. Always cite your sources using [Source X] format."""
        
        return prompt
    
    def _extract_citations(self, answer: str, sources: List[Dict]) -> List[Dict]:
        """Extract citations from the answer and match them to sources"""
        # Find all [Source X] patterns
        citations = re.findall(r'\[Source (\d+)\]', answer)
        unique_citations = list(set([int(c) for c in citations]))
        
        # Match citations to sources
        cited_sources = []
        for idx in unique_citations:
            if 1 <= idx <= len(sources):
                source = sources[idx - 1].copy()
                source['citation_number'] = idx
                cited_sources.append(source)
        
        return cited_sources
    
    def answer_with_retrieval(self, query: str, retrieval_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer from retrieval results
        
        Args:
            query: User's question
            retrieval_results: Results from hybrid search or web search
            
        Returns:
            Generated answer with sources
        """
        context = retrieval_results.get('context', '')
        sources = retrieval_results.get('sources', [])
        
        return self.generate_answer(query, context, sources)
    
    def is_available(self) -> bool:
        """Check if the answer generator is available"""
        return self.client is not None


# Test function
def test_answer_generator():
    """Test the answer generator"""
    print("=" * 60)
    print("🧪 BIT 5: Testing Answer Generator")
    print("=" * 60)
    
    # Check if API key is set
    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("⚠️ GROQ_API_KEY not set in .env file")
        print("   Please add: GROQ_API_KEY=your_api_key")
        print("   Skipping test...")
        return
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Initialize generator
    generator = AnswerGenerator()
    
    if not generator.is_available():
        print("❌ Failed to initialize Groq client")
        return
    
    # Test with sample context
    test_sources = [
        {
            'content': 'Machine learning is a subset of artificial intelligence that enables systems to learn from data.',
            'source': 'doc1.txt',
            'source_type': 'document',
            'title': 'Machine Learning Basics'
        },
        {
            'content': 'RAG stands for Retrieval-Augmented Generation, combining retrieval and generation for better answers.',
            'source': 'doc2.txt',
            'source_type': 'document',
            'title': 'RAG Explained'
        },
        {
            'content': 'Python is a versatile programming language with simple syntax and powerful libraries for data science.',
            'source': 'doc3.txt',
            'source_type': 'document',
            'title': 'Python Programming'
        }
    ]
    
    test_queries = [
        "What is machine learning?",
        "Explain RAG and how it works",
        "What programming language is good for data science?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        # Build context from sources
        context = "\n".join([s['content'] for s in test_sources])
        
        # Generate answer
        result = generator.generate_answer(query, context, test_sources)
        
        if result['error']:
            print(f"❌ Error: {result['answer']}")
        else:
            print(f"💬 Answer:\n{result['answer']}")
            print(f"\n📚 Sources cited: {len(result['cited_sources'])}")
            if result['cited_sources']:
                for source in result['cited_sources'][:2]:
                    print(f"   - [Source {source.get('citation_number', '?')}] {source.get('source', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("✅ BIT 5: Answer Generator Module COMPLETE!")
    print("=" * 60)
    print("\n📋 Checklist:")
    print("   ✅ Groq integration")
    print("   ✅ Prompt engineering")
    print("   ✅ Source citation")
    print("   ✅ Error handling")


if __name__ == "__main__":
    test_answer_generator()