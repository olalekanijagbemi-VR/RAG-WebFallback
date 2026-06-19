"""
Vector Store Module
Handles FAISS vector store with fastembed (lightweight, no PyTorch issues)
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from groq import Groq

# Try to import fastembed
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
    print("✅ Fastembed available")
except ImportError:
    FASTEMBED_AVAILABLE = False
    print("⚠️ Fastembed not available, using fallback")

class SimpleEmbeddings:
    """
    Simple embeddings class using fastembed or fallback
    """
    
    def __init__(self):
        self.model = None
        self.model_name = "all-MiniLM-L6-v2"
        
        if FASTEMBED_AVAILABLE:
            try:
                self.model = TextEmbedding(model_name=self.model_name)
                print(f"✅ Fastembed model loaded: {self.model_name}")
            except Exception as e:
                print(f"⚠️ Error loading fastembed: {e}")
                self.model = None
        else:
            print("⚠️ Fastembed not available, using fallback embeddings")
            self._init_fallback()
        
        # If fastembed failed or not available, use a simple fallback
        if self.model is None and not hasattr(self, 'fallback_mode'):
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback embeddings using numpy random (for testing only)"""
        self.fallback_mode = True
        print("⚠️ Using FALLBACK embeddings (random vectors - for testing only)")
        print("⚠️ For production, install: pip install fastembed")
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """Generate a deterministic but random-looking embedding for testing"""
        # Use hash of text to generate consistent embeddings
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % 2**32)
        # Generate 384-dim vector (common for MiniLM)
        embedding = np.random.randn(384).astype(np.float32)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents"""
        if hasattr(self, 'fallback_mode') and self.fallback_mode:
            return [self._get_fallback_embedding(text) for text in texts]
        
        if self.model is not None:
            # Use fastembed
            embeddings = list(self.model.embed(texts))
            return [emb.tolist() for emb in embeddings]
        else:
            # Fallback
            return [self._get_fallback_embedding(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query"""
        if hasattr(self, 'fallback_mode') and self.fallback_mode:
            return self._get_fallback_embedding(text)
        
        if self.model is not None:
            # Use fastembed
            embeddings = list(self.model.embed([text]))
            return embeddings[0].tolist()
        else:
            return self._get_fallback_embedding(text)

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self):
        self.embeddings = SimpleEmbeddings()
        self.vector_store = None
        self.store_path = None
        self.groq_client = None
        
        # Initialize Groq client if API key available
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            try:
                self.groq_client = Groq(api_key=api_key)
                print("✅ Groq client initialized")
            except Exception as e:
                print(f"⚠️ Groq client init failed: {e}")
        
        print("✅ VectorStoreManager initialized")
        
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from documents
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"📊 Creating vector store with {len(documents)} documents...")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        print(f"✅ Vector store created with {self.vector_store.index.ntotal} vectors")
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add documents to existing vector store
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call create_vector_store first.")
        
        try:
            self.vector_store.add_documents(documents)
            print(f"✅ Added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search the vector store for relevant documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            print(f"🔍 Search returned {len(results)} results")
            return results
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def search_with_score(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search and return formatted results with metadata
        """
        results = self.search(query, k)
        formatted_results = []
        
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score),
                'source': doc.metadata.get('source_file', 'Unknown')
            })
        
        return formatted_results
    
    def save(self, path: str) -> bool:
        """
        Save vector store to disk
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        try:
            self.store_path = path
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            self.vector_store.save_local(path)
            print(f"💾 Vector store saved to {path}")
            return True
        except Exception as e:
            st.error(f"Error saving vector store: {str(e)}")
            return False
    
    def load(self, path: str) -> bool:
        """
        Load vector store from disk
        """
        try:
            self.store_path = path
            self.vector_store = FAISS.load_local(
                path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"📂 Vector store loaded from {path}")
            return True
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if not self.vector_store:
            return {"status": "Not initialized"}
        
        # Get index statistics
        index = self.vector_store.index
        return {
            "status": "Initialized",
            "total_vectors": index.ntotal if hasattr(index, 'ntotal') else 0,
            "dimension": index.d if hasattr(index, 'd') else 0,
            "store_path": self.store_path
        }
    
    def clear(self):
        """Clear the vector store"""
        self.vector_store = None
        self.store_path = None
        print("🧹 Vector store cleared")

# Utility function for quick testing
def test_vector_store():
    """Quick test function"""
    print("🧪 Testing Vector Store Module...")
    print("-" * 40)
    
    # Create test documents
    sample_texts = [
        "This is a test document about machine learning.",
        "FAISS is a library for efficient similarity search.",
        "Groq provides fast LLM inference.",
        "Vector embeddings convert text to numbers.",
        "RAG systems combine retrieval and generation.",
        "Streamlit is great for building AI apps.",
        "Python is a versatile programming language.",
        "Document processing involves loading and chunking text."
    ]
    
    test_docs = []
    for i, text in enumerate(sample_texts):
        doc = Document(
            page_content=text,
            metadata={"source_file": f"test_doc_{i}.txt", "index": i}
        )
        test_docs.append(doc)
    
    print(f"📄 Created {len(test_docs)} test documents")
    
    # Initialize vector store
    try:
        vstore = VectorStoreManager()
        vstore.create_vector_store(test_docs)
        
        print(f"✅ Vector Store initialized")
        stats = vstore.get_stats()
        print(f"📊 Total vectors: {stats['total_vectors']}")
        print(f"📊 Dimension: {stats['dimension']}")
        
        # Test search
        print("\n🔍 Testing search functionality...")
        results = vstore.search("machine learning", k=3)
        print(f"🔍 Search returned {len(results)} results")
        for i, (doc, score) in enumerate(results, 1):
            print(f"   {i}. Score: {score:.4f}: {doc.page_content[:50]}...")
        
        # Test formatted search
        print("\n🔍 Testing formatted search...")
        formatted = vstore.search_with_score("FAISS library", k=2)
        for i, result in enumerate(formatted, 1):
            print(f"   {i}. Source: {result['source']}")
            print(f"      Score: {result['score']:.4f}")
            print(f"      Content: {result['content'][:50]}...")
        
        print("\n" + "-" * 40)
        print("✅ Vector Store Module ready for integration!")
        return vstore
        
    except Exception as e:
        print(f"❌ Error in test: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_vector_store()