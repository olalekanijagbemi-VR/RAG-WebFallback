"""
Quick test for document processor
Run: python test_doc_processor.py
"""

from src.document_processor import DocumentProcessor

def test():
    processor = DocumentProcessor()
    print("✅ Document Processor loaded successfully!")
    print(f"📄 Supported formats: {processor.supported_formats}")
    print(f"📊 Chunk size: {processor.text_splitter._chunk_size}")
    print(f"📊 Chunk overlap: {processor.text_splitter._chunk_overlap}")
    print("\n✅ Step 2 ready for integration!")

if __name__ == "__main__":
    test()