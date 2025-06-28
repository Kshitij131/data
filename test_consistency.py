from gpt_agent import GPTAgent
import time

def test_consistency():
    """Test if the analyze_file function gives consistent results for the same file"""
    # Initialize the agent
    agent = GPTAgent()
    
    # Test file
    file_path = "Clients_Clean_Consistent.csv"
    
    # First analysis
    print("\n=== First Analysis ===")
    analysis1 = agent.analyze_file(file_path)
    
    # Get issue types and counts
    issue_types1 = set()
    issue_locations1 = set()
    
    if analysis1["status"] == "success" and "detailed_analysis" in analysis1 and "issues" in analysis1["detailed_analysis"]:
        for issue in analysis1["detailed_analysis"]["issues"]:
            issue_type = issue.get("type", "unknown")
            issue_location = issue.get("location", "unknown")
            issue_types1.add(issue_type)
            issue_locations1.add(f"{issue_type}:{issue_location}")
    
    # Wait a moment before second analysis
    print("\nWaiting 3 seconds before second analysis...")
    time.sleep(3)
    
    # Second analysis
    print("\n=== Second Analysis ===")
    analysis2 = agent.analyze_file(file_path)
    
    # Get issue types and counts
    issue_types2 = set()
    issue_locations2 = set()
    
    if analysis2["status"] == "success" and "detailed_analysis" in analysis2 and "issues" in analysis2["detailed_analysis"]:
        for issue in analysis2["detailed_analysis"]["issues"]:
            issue_type = issue.get("type", "unknown")
            issue_location = issue.get("location", "unknown")
            issue_types2.add(issue_type)
            issue_locations2.add(f"{issue_type}:{issue_location}")
    
    # Compare results
    print("\n=== Comparison ===")
    print(f"First analysis found issues of types: {issue_types1}")
    print(f"Second analysis found issues of types: {issue_types2}")
    
    # Check if the issue types match
    types_match = issue_types1 == issue_types2
    
    # Check if the issue locations match
    locations_match = issue_locations1 == issue_locations2
    
    print(f"\nIssue types match: {types_match}")
    print(f"Issue locations match: {locations_match}")
    
    if types_match and locations_match:
        print("\n✅ PASS: The analyze_file function gives consistent results")
    else:
        print("\n❌ FAIL: The analyze_file function gives inconsistent results")
        print("\nDifferences in types:", issue_types1.symmetric_difference(issue_types2))
        print("Differences in locations:", issue_locations1.symmetric_difference(issue_locations2))

if __name__ == "__main__":
    test_consistency()
