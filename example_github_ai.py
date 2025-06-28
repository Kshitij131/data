"""
Example script demonstrating how to use the GPT agent with GitHub AI endpoint
using the OpenAI SDK directly.
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Get GitHub token from environment
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("❌ GitHub token not found. Please set GITHUB_TOKEN in your .env file.")
        return
    
    # GitHub AI endpoint and model
    endpoint = "https://models.github.ai/inference"
    model = "meta-llama/llama-3.1-70b-instruct"  # Try with Llama model
    
    try:
        # Initialize OpenAI client with GitHub AI configuration
        client = OpenAI(
            base_url=endpoint,
            api_key=token
        )
        
        # Make a simple request
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": "What is the capital of France?",
                }
            ],
            temperature=1.0,
            top_p=1.0
        )
        
        # Print the response
        print("\n=== GitHub AI Response ===")
        print(response.choices[0].message.content)
        print("\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This could be due to invalid token or no access to the specified model.")
        
        # Try with standard OpenAI as fallback
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and not openai_key.startswith("sk-dummy"):
            try:
                print("\nTrying with standard OpenAI as fallback...")
                client = OpenAI(api_key=openai_key)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Use a widely available model
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": "What is the capital of France?",
                        }
                    ],
                    temperature=1.0
                )
                print("\n=== Standard OpenAI Response ===")
                print(response.choices[0].message.content)
            except Exception as e2:
                print(f"❌ Error with OpenAI fallback: {e2}")

if __name__ == "__main__":
    main()
