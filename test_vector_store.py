"""
Quick test for vector store
Run: python test_vector_store.py
"""

from src.vector_store import test_vector_store

def test():
    print("🧪 Testing Vector Store Module...")
    print("-" * 40)
    vstore = test_vector_store()
    print("-" * 40)
    print("✅ Vector Store Module ready for integration!")

if __name__ == "__main__":
    test()