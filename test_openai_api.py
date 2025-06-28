"""
Test script specifically for OpenAI API integration
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

def test_openai_api():
    """Test OpenAI API directly"""
    print("=== Testing OpenAI API ===")
    
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found in environment variables")
        return False
    
    # Get OpenAI model
    model = os.getenv("OPENAI_MODEL", "gpt-4o-2024-05-13")
    print(f"Model: {model}")
    
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        test_prompt = "What is the capital of France? Answer in one word."
        
        print(f"\nSending test prompt: '{test_prompt}'")
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print("✅ OpenAI API is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing OpenAI API: {e}")
        return False

if __name__ == "__main__":
    test_openai_api()
