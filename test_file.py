from gpt_agent import GPTAgent

# Initialize the agent
agent = GPTAgent()

# Test file analysis
print("Testing File Analysis for Clients_Clean_Consistent.csv")
analysis = agent.analyze_file("Clients_Clean_Consistent.csv")

# Print the results
print(f"Analysis Status: {analysis['status']}")
if analysis['status'] == 'success':
    print(f"File: {analysis['file_info']['file_name']}")
    print(f"Type: {analysis['file_info']['file_type']}")
    print(f"Rows: {analysis['file_info']['rows']}")
    print(f"Columns: {analysis['file_info']['columns']}")
    
    # Print summary information
    print(f"\nFound {analysis['summary']['total_issues']} issues:")
    
    # Print detailed issues if found
    if 'detailed_analysis' in analysis and 'issues' in analysis['detailed_analysis']:
        print("\nIssues found:")
        for i, issue in enumerate(analysis['detailed_analysis']['issues']):
            print(f"{i+1}. {issue.get('type', 'Unknown')} - {issue.get('description', 'No description')}")
            print(f"   Location: {issue.get('location', 'Unknown')}")
            print(f"   Fix: {issue.get('fix', 'No fix recommendation')}")
            print()
    else:
        print("No detailed issues found.")
else:
    print(f"Error: {analysis.get('message', 'Unknown error')}")
