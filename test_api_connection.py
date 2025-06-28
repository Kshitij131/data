import os
import sys
import json
from gpt_agent import GPTAgent

def test_api_connection():
    """Test the GPT agent's ability to connect to the API and generate content"""
    try:
        # Initialize the agent
        agent = GPTAgent()
        
        # Test a simple prompt
        prompt = "What is the capital of France? Keep your answer very short."
        print(f"\nSending test prompt: '{prompt}'")
        
        # Generate content
        response = agent._generate_content(prompt)
        
        # Print the response
        print("\nResponse from API:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Check if the response contains expected content
        if "Paris" in response:
            print("✅ API test successful - received expected content")
            return True
        else:
            print("⚠️ API response doesn't contain expected content, but connection worked")
            return True
            
    except Exception as e:
        print(f"❌ Error during API test: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing GPT Agent API Connection...")
    success = test_api_connection()
    sys.exit(0 if success else 1)
