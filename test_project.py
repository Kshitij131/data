"""
Consolidated test script for the Data Alchemist project.
Tests all essential functionality: GitHub AI connection, file analysis, and fixes.
"""
import os
import json
import sys
from dotenv import load_dotenv
from gpt_agent import GPTAgent
import pandas as pd

def check_environment():
    """Check if the environment is properly configured"""
    print("=== Environment Check ===")
    load_dotenv()
    
    # Check GitHub token
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("❌ GITHUB_TOKEN not found in environment")
        return False
    print("✅ GITHUB_TOKEN found")
    
    # Check GitHub AI endpoint
    github_ai_endpoint = os.getenv("GITHUB_AI_ENDPOINT")
    if not github_ai_endpoint:
        print("❌ GITHUB_AI_ENDPOINT not found in environment")
        return False
    print(f"✅ GITHUB_AI_ENDPOINT: {github_ai_endpoint}")
    
    # Check GitHub AI model
    github_ai_model = os.getenv("GITHUB_AI_MODEL")
    if not github_ai_model:
        print("❌ GITHUB_AI_MODEL not found in environment")
        return False
    print(f"✅ GITHUB_AI_MODEL: {github_ai_model}")
    
    return True

def test_api_connection():
    """Test the basic connection to the GitHub AI API"""
    print("\n=== Testing API Connection ===")
    
    try:
        # Initialize agent
        agent = GPTAgent()
        
        # Simple test prompt
        prompt = "What is the capital of France? Keep your answer very short."
        print(f"Sending test prompt: '{prompt}'")
        
        # Generate response
        response = agent._generate_content(prompt)
        
        print(f"Response: {response}")
        
        if "Paris" in response:
            print("✅ API test successful - received expected content")
            return True
        else:
            print("⚠️ API response doesn't contain expected content, but connection worked")
            return True
    
    except Exception as e:
        print(f"❌ Error during API test: {str(e)}")
        return False

def test_file_analysis():
    """Test the file analysis functionality"""
    print("\n=== Testing File Analysis ===")
    
    # Check if test file exists
    file_path = "Clients_Sample_Inconsistent.csv"
    if not os.path.exists(file_path):
        print(f"❌ Test file not found: {file_path}")
        return False
    
    try:
        # Initialize agent
        agent = GPTAgent()
        
        # Analyze file
        print(f"Analyzing file: {file_path}")
        analysis = agent.analyze_file(file_path)
        
        # Print summary of analysis
        if analysis["status"] == "success":
            print(f"✅ File analysis successful")
            
            # Print summary information
            file_info = analysis.get("file_info", {})
            print(f"File: {file_info.get('file_name')}")
            print(f"Type: {file_info.get('file_type')}")
            print(f"Rows: {file_info.get('rows')}")
            print(f"Columns: {file_info.get('columns')}")
            
            # Print issues summary
            summary = analysis.get("summary", {})
            total_issues = summary.get("total_issues", 0)
            print(f"Found {total_issues} issues")
            
            # Print issue types
            issue_types = summary.get("issue_types", {})
            for issue_type, count in issue_types.items():
                print(f"- {issue_type}: {count}")
            
            return True
        else:
            print(f"❌ File analysis failed: {analysis.get('message', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"❌ Error during file analysis: {str(e)}")
        return False

def test_file_fix():
    """Test the file fixing functionality"""
    print("\n=== Testing File Fixing ===")
    
    # Check if test file exists
    file_path = "Clients_Sample_Inconsistent.csv"
    if not os.path.exists(file_path):
        print(f"❌ Test file not found: {file_path}")
        return False
    
    try:
        # Initialize agent
        agent = GPTAgent()
        
        # Fix file
        print(f"Fixing file: {file_path}")
        fix_result = agent.fix_file_errors(file_path)
        
        # Print summary of fixes
        if fix_result["status"] == "success":
            print(f"✅ File fix successful")
            
            # Print applied fixes
            fixes_applied = fix_result.get("fixes_applied", [])
            print(f"Applied {len(fixes_applied)} fixes:")
            for i, fix in enumerate(fixes_applied):
                print(f"- Fix {i+1}: {fix.get('issue_type')} in {fix.get('location')}")
            
            # Print failed fixes
            fixes_failed = fix_result.get("fixes_failed", [])
            if fixes_failed:
                print(f"Failed to apply {len(fixes_failed)} fixes:")
                for i, fail in enumerate(fixes_failed):
                    print(f"- {i+1}: {fail.get('reason')}")
            
            # Print metrics
            metrics = fix_result.get("metrics", {})
            before = metrics.get("before", {})
            after = metrics.get("after", {})
            
            print("\nBefore vs After:")
            print(f"- Rows: {before.get('rows')} → {after.get('rows')}")
            
            fixed_file = fix_result.get("fixed_file")
            if fixed_file and os.path.exists(fixed_file):
                print(f"Fixed file created: {fixed_file}")
            
            return True
        else:
            print(f"❌ File fix failed: {fix_result.get('message', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"❌ Error during file fix: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=== Data Alchemist Project Test ===")
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return
    
    # Run tests
    api_test = test_api_connection()
    if not api_test:
        print("❌ API connection test failed")
        return
    
    analysis_test = test_file_analysis()
    if not analysis_test:
        print("❌ File analysis test failed")
    
    fix_test = test_file_fix()
    if not fix_test:
        print("❌ File fix test failed")
    
    if api_test and analysis_test and fix_test:
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️ Some tests failed")

if __name__ == "__main__":
    main()