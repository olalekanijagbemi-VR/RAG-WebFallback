"""
Test Bit 3: Router Agent
Run: python test_bit3.py
"""

from src.router_agent import test_router_agent

def test_bit3():
    print("\n" + "=" * 60)
    print("🧪 Running BIT 3 Test")
    print("=" * 60)
    
    test_router_agent()
    
    print("\n" + "=" * 60)
    print("✅ BIT 3 Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_bit3()