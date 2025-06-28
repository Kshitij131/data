"""
Test script specifically for GitHub AI integration via Azure SDK
"""
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

def test_github_ai():
    """Test GitHub AI with Azure SDK directly"""
    print("=== Testing GitHub AI with Azure SDK ===")
    
    # Load environment variables
    load_dotenv()
    
    # Get GitHub AI credentials
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ GitHub token not found in environment variables")
        return False
    
    # Get GitHub AI endpoint and model
    github_ai_endpoint = os.getenv("GITHUB_AI_ENDPOINT", "https://models.github.ai/inference")
    github_ai_model = os.getenv("GITHUB_AI_MODEL", "openai/gpt-4.1")
    
    print(f"Endpoint: {github_ai_endpoint}")
    print(f"Model: {github_ai_model}")
    
    try:
        # Initialize the Azure AI Inference client
        client = ChatCompletionsClient(
            endpoint=github_ai_endpoint,
            credential=AzureKeyCredential(github_token)
        )
        
        # Test with a simple request
        test_prompt = "What is the capital of France? Answer in one word."
        
        print(f"\nSending test prompt: '{test_prompt}'")
        
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content=test_prompt)
            ],
            temperature=0.1,
            max_tokens=50,
            model=github_ai_model
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print("✅ GitHub AI is working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Error testing GitHub AI: {e}")
        return False

if __name__ == "__main__":
    test_github_ai()
