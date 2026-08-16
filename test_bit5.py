"""
Test Bit 5: Answer Generator
Run: python test_bit5.py
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from src.answer_generator import test_answer_generator

def test_bit5():
    print("\n" + "=" * 60)
    print("🧪 Running BIT 5 Test")
    print("=" * 60)
    
    # Check if API key is loaded
    api_key = os.getenv('GROQ_API_KEY')
    print(f"📝 API Key loaded: {'✅ Yes' if api_key else '❌ No'}")
    if api_key:
        print(f"   API Key starts with: {api_key[:10]}...")
    
    test_answer_generator()
    
    print("\n" + "=" * 60)
    print("✅ BIT 5 Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_bit5()