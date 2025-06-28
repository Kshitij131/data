"""
Comprehensive test script for the GPT agent 
Tests both Azure SDK for GitHub AI and OpenAI API functionality
"""
import os
import json
import time
import csv
from dotenv import load_dotenv
from gpt_agent import GPTAgent, GITHUB_AI_MODELS

def test_generate_content(agent):
    """Test the basic content generation functionality"""
    print("\n=== Testing Basic Content Generation ===")
    test_prompt = "What is the capital of France? Give a brief answer."
    
    try:
        start_time = time.time()
        result = agent._generate_content(test_prompt)
        duration = time.time() - start_time
        print(f"Response: {result}")
        print(f"Duration: {duration:.2f} seconds")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_header_mapping(agent):
    """Test the header mapping functionality"""
    print("\n=== Testing Header Mapping ===")
    test_headers = [
        "client_id",
        "Priority Level",
        "client name",
        "GroupTags",
        "attribute_json"
    ]
    
    results = {}
    for header in test_headers:
        try:
            mapped = agent.map_header(header)
            results[header] = mapped
            print(f"'{header}' → '{mapped}'")
        except Exception as e:
            print(f"❌ Error mapping '{header}': {e}")
            results[header] = f"ERROR: {str(e)}"
    
    return results

def test_validation(agent):
    """Test the data validation functionality"""
    print("\n=== Testing Data Validation ===")
    
    # Load sample data from CSV
    try:
        with open('Clients_Sample_Inconsistent.csv', 'r') as f:
            csv_reader = csv.DictReader(f)
            test_data = [row for row in csv_reader]
            
        # Use only first 5 rows for testing
        test_data = test_data[:5]
        
        start_time = time.time()
        validation_result = agent.validate_data(test_data)
        duration = time.time() - start_time
        
        print(f"Found {len(validation_result.get('errors', []))} validation errors")
        for i, error in enumerate(validation_result.get('errors', [])[:3]):
            print(f"Error {i+1}: {error.get('type')} - {error.get('message')}")
        
        print(f"Duration: {duration:.2f} seconds")
        return validation_result
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return None

def test_nl_to_filter(agent):
    """Test natural language to filter conversion"""
    print("\n=== Testing Natural Language to Filter ===")
    
    test_queries = [
        "Find clients with priority level higher than 3",
        "Show tasks with Python in required skills"
    ]
    
    results = {}
    for query in test_queries:
        try:
            start_time = time.time()
            filter_result = agent.query_to_filter(query)
            duration = time.time() - start_time
            
            print(f"\nQuery: '{query}'")
            print(f"Filter: {json.dumps(filter_result, indent=2)}")
            print(f"Duration: {duration:.2f} seconds")
            
            results[query] = filter_result
        except Exception as e:
            print(f"❌ Error converting query '{query}': {e}")
            results[query] = f"ERROR: {str(e)}"
    
    return results

def test_rule_generation(agent):
    """Test natural language to rule conversion"""
    print("\n=== Testing Rule Generation ===")
    
    test_rules = [
        "Tasks T1 and T2 must run together in the morning phase",
        "Workers with Python skills can handle maximum 3 tasks per phase"
    ]
    
    results = {}
    for rule in test_rules:
        try:
            start_time = time.time()
            rule_result = agent.nl_to_rule(rule)
            duration = time.time() - start_time
            
            print(f"\nRule: '{rule}'")
            print(f"Result: {json.dumps(rule_result, indent=2)}")
            print(f"Duration: {duration:.2f} seconds")
            
            results[rule] = rule_result
        except Exception as e:
            print(f"❌ Error generating rule from '{rule}': {e}")
            results[rule] = f"ERROR: {str(e)}"
    
    return results

def main():
    # Load environment variables with force reload
    load_dotenv(override=True)
    
    # Print configuration information
    print("=== GPT Agent Test Configuration ===")
    print(f"GitHub Token: {'Configured' if os.getenv('GITHUB_TOKEN') else 'Not configured'}")
    print(f"OpenAI API Key: {'Configured' if os.getenv('OPENAI_API_KEY') else 'Not configured'}")
    print(f"GitHub AI Endpoint: {os.getenv('GITHUB_AI_ENDPOINT', 'Not configured')}")
    print(f"GitHub AI Model: {os.getenv('GITHUB_AI_MODEL', 'Not configured')}")
    print(f"OpenAI Model: {os.getenv('OPENAI_MODEL', 'Not configured')}")
    
    # Initialize the GPT agent
    print("\nInitializing GPT agent...")
    try:
        agent = GPTAgent()
        
        # Run tests
        tests = {
            "Basic Content Generation": test_generate_content,
            "Header Mapping": test_header_mapping,
            "Data Validation": test_validation,
            "Natural Language to Filter": test_nl_to_filter,
            "Rule Generation": test_rule_generation
        }
        
        results = {}
        
        print("\n=== Running Tests ===")
        for test_name, test_func in tests.items():
            print(f"\n🔍 Running test: {test_name}")
            try:
                result = test_func(agent)
                success = result is not None
                results[test_name] = "✅ Success" if success else "❌ Failed"
            except Exception as e:
                print(f"❌ Test '{test_name}' failed with error: {e}")
                results[test_name] = f"❌ Error: {str(e)}"
        
        # Print summary
        print("\n=== Test Summary ===")
        for test_name, status in results.items():
            print(f"{test_name}: {status}")
            
    except Exception as e:
        print(f"❌ Failed to initialize GPT agent: {e}")

if __name__ == "__main__":
    main()
