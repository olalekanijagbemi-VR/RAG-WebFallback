"""
Hybrid Search Module - Bit 2 (FIXED VERSION)
Combines FAISS (semantic) + BM25 (keyword) for optimal retrieval
"""

import os
import sys
import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# BM25 for keyword search
from rank_bm25 import BM25Okapi

class HybridSearch:
    """
    Hybrid search combining FAISS (semantic) and BM25 (keyword)
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize hybrid search
        
        Args:
            embedding_model: HuggingFace embedding model name
        """
        print("🔄 Initializing Hybrid Search...")
        
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Embeddings loaded: {embedding_model}")
        except Exception as e:
            print(f"⚠️ Error loading embeddings: {e}")
            print("   Trying alternative loading method...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model
            )
            print(f"✅ Embeddings loaded (alternative method): {embedding_model}")
        
        # Initialize components
        self.vectorstore = None
        self.bm25 = None
        self.documents = []
        self.chunk_texts = []
        self.initialized = False
        
        # Search weights
        self.semantic_weight = 0.6
        self.keyword_weight = 0.4
        
    def build_index(self, documents: List[Document]) -> bool:
        """
        Build both FAISS and BM25 indices from documents
        """
        if not documents:
            print("❌ No documents provided to build index")
            return False
        
        try:
            print(f"📊 Building hybrid index with {len(documents)} chunks...")
            
            # Store documents and extract text
            self.documents = documents
            self.chunk_texts = [doc.page_content for doc in documents]
            
            # 1. Build FAISS (semantic) index
            print("   🔨 Building FAISS index...")
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            print(f"   ✅ FAISS index built with {self.vectorstore.index.ntotal} vectors")
            
            # 2. Build BM25 (keyword) index
            print("   🔨 Building BM25 index...")
            tokenized_corpus = [self._tokenize(text) for text in self.chunk_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"   ✅ BM25 index built with {len(self.chunk_texts)} documents")
            
            self.initialized = True
            print(f"✅ Hybrid index built successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error building hybrid index: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search"""
        if not self.initialized:
            print("⚠️ Search index not initialized")
            return []
        
        try:
            # 1. Semantic search with FAISS
            semantic_results = self._semantic_search(query, k * 2)
            
            # 2. Keyword search with BM25
            keyword_results = self._keyword_search(query, k * 2)
            
            # 3. Combine and re-rank results
            combined_results = self._combine_results(
                semantic_results, 
                keyword_results, 
                k
            )
            
            return combined_results
            
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform semantic search using FAISS"""
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                try:
                    idx = self.documents.index(doc)
                except ValueError:
                    idx = -1
                
                confidence = 1.0 / (1.0 + score)
                
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'confidence': float(confidence),
                    'index': idx,
                    'search_type': 'semantic'
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"⚠️ Semantic search error: {e}")
            return []
    
    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Perform keyword search using BM25"""
        try:
            tokenized_query = self._tokenize(query)
            scores = self.bm25.get_scores(tokenized_query)
            
            top_indices = np.argsort(scores)[::-1][:k]
            
            formatted_results = []
            for idx in top_indices:
                score = scores[idx]
                if score > 0:
                    confidence = min(1.0, score / 10.0)
                    doc = self.documents[idx]
                    
                    formatted_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': float(score),
                        'confidence': float(confidence),
                        'index': idx,
                        'search_type': 'keyword'
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"⚠️ Keyword search error: {e}")
            return []
    
    def _combine_results(self, semantic_results: List[Dict], 
                         keyword_results: List[Dict], 
                         k: int) -> List[Dict[str, Any]]:
        """Combine and re-rank results"""
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            idx = result['index']
            if idx != -1:
                if idx not in combined:
                    combined[idx] = {
                        'content': result['content'],
                        'metadata': result['metadata'],
                        'semantic_score': result['confidence'],
                        'keyword_score': 0.0,
                        'semantic_raw': result['score'],
                        'keyword_raw': 0.0
                    }
                else:
                    combined[idx]['semantic_score'] = max(
                        combined[idx]['semantic_score'], 
                        result['confidence']
                    )
                    combined[idx]['semantic_raw'] = min(
                        combined[idx]['semantic_raw'], 
                        result['score']
                    )
        
        # Add keyword results
        for result in keyword_results:
            idx = result['index']
            if idx != -1:
                if idx not in combined:
                    combined[idx] = {
                        'content': result['content'],
                        'metadata': result['metadata'],
                        'semantic_score': 0.0,
                        'keyword_score': result['confidence'],
                        'semantic_raw': 999.0,
                        'keyword_raw': result['score']
                    }
                else:
                    combined[idx]['keyword_score'] = max(
                        combined[idx]['keyword_score'], 
                        result['confidence']
                    )
                    combined[idx]['keyword_raw'] = max(
                        combined[idx]['keyword_raw'], 
                        result['score']
                    )
        
        # Calculate combined score
        for idx in combined:
            sem = combined[idx]['semantic_score']
            kw = combined[idx]['keyword_score']
            
            combined[idx]['combined_score'] = (
                self.semantic_weight * sem + 
                self.keyword_weight * kw
            )
            
            if sem > kw:
                combined[idx]['primary_search'] = 'semantic'
                combined[idx]['confidence'] = sem
            else:
                combined[idx]['primary_search'] = 'keyword'
                combined[idx]['confidence'] = kw
        
        # Sort by combined score
        sorted_results = sorted(
            combined.values(), 
            key=lambda x: x['combined_score'], 
            reverse=True
        )
        
        # Return top k
        top_results = []
        for result in sorted_results[:k]:
            top_results.append({
                'content': result['content'],
                'metadata': result['metadata'],
                'confidence': result['confidence'],
                'combined_score': result['combined_score'],
                'semantic_score': result['semantic_score'],
                'keyword_score': result['keyword_score'],
                'primary_search': result['primary_search']
            })
        
        return top_results
    
    def search_with_context(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search and return results with context"""
        results = self.search(query, k)
        
        if not results:
            return {
                'results': [],
                'context': "",
                'sources': [],
                'confidence': 0.0
            }
        
        context_parts = []
        sources = []
        total_confidence = 0.0
        
        for i, result in enumerate(results):
            context_parts.append(f"[Source {i+1}]: {result['content']}")
            
            source_info = {
                'content': result['content'],
                'source': result['metadata'].get('source_file', 'Unknown'),
                'confidence': result['confidence'],
                'search_type': result['primary_search']
            }
            sources.append(source_info)
            total_confidence += result['confidence']
        
        avg_confidence = total_confidence / len(results) if results else 0.0
        
        return {
            'results': results,
            'context': "\n\n".join(context_parts),
            'sources': sources,
            'confidence': avg_confidence,
            'num_results': len(results)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the hybrid search index"""
        if not self.initialized:
            return {"status": "Not initialized"}
        
        return {
            "status": "Initialized",
            "num_documents": len(self.documents),
            "vectorstore_type": "FAISS",
            "keyword_index_type": "BM25",
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight
        }


# Test function
def test_hybrid_search():
    """Test the hybrid search module"""
    print("=" * 60)
    print("🧪 BIT 2: Testing Hybrid Search")
    print("=" * 60)
    
    # Create sample documents
    sample_docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            metadata={"source_file": "doc1.txt"}
        ),
        Document(
            page_content="FAISS is a library for efficient similarity search and clustering of dense vectors.",
            metadata={"source_file": "doc2.txt"}
        ),
        Document(
            page_content="Natural language processing helps computers understand human language.",
            metadata={"source_file": "doc3.txt"}
        ),
        Document(
            page_content="Python is a programming language with simple syntax and powerful libraries.",
            metadata={"source_file": "doc4.txt"}
        ),
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation, combining retrieval and generation.",
            metadata={"source_file": "doc5.txt"}
        )
    ]
    
    print(f"📄 Created {len(sample_docs)} test documents")
    
    # Initialize hybrid search
    search_engine = HybridSearch()
    
    # Build index
    success = search_engine.build_index(sample_docs)
    if not success:
        print("❌ Failed to build index")
        return
    
    print("\n" + "=" * 60)
    print("🔍 Testing Search Queries")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "FAISS library",
        "Python programming",
        "RAG system"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        results = search_engine.search(query, k=3)
        
        if not results:
            print("  No results found")
            continue
            
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Content: {result['content'][:80]}...")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Primary Search: {result['primary_search']}")
            print(f"  Semantic Score: {result['semantic_score']:.3f}")
            print(f"  Keyword Score: {result['keyword_score']:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ BIT 2: Hybrid Search Module COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_hybrid_search()