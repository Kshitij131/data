from gpt_agent import GPTAgent

# Initialize the agent
agent = GPTAgent()

# Test file analysis and fixing
print("Testing File Analysis and Fixing for Clients_Clean_Consistent.csv")
analysis = agent.analyze_file("Clients_Clean_Consistent.csv")

# Print the analysis results
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
    
    # Only attempt to fix if there are issues
    if analysis['summary']['total_issues'] > 0:
        print("\n=== Testing File Fix ===")
        fix_result = agent.fix_file_errors("Clients_Clean_Consistent.csv")
        
        print(f"Fix Status: {fix_result['status']}")
        print(f"Message: {fix_result['message']}")
        
        if fix_result['status'] == 'success' and fix_result['fixed_file']:
            print(f"Fixed file created: {fix_result['fixed_file']}")
            print(f"\nApplied {len(fix_result['fixes_applied'])} fixes:")
            
            # Display details of applied fixes
            for i, fix in enumerate(fix_result['fixes_applied']):
                print(f"\nFix {i+1}: {fix.get('issue_type', 'Unknown issue')}")
                print(f"  Location: {fix.get('location', 'Unknown')}")
                print(f"  Description: {fix.get('description', 'No description')}")
            
            # Show failed fixes if any
            if fix_result['fixes_failed']:
                print(f"\nFailed to apply {len(fix_result['fixes_failed'])} fixes:")
                for i, fail in enumerate(fix_result['fixes_failed']):
                    print(f"  {i+1}. {fail.get('issue', {}).get('type', 'Unknown')} - {fail.get('reason', 'Unknown reason')}")
            
            # Show metrics
            if 'metrics' in fix_result:
                print("\nBefore vs After Metrics:")
                metrics = fix_result['metrics']
                print(f"  Rows: {metrics['before']['rows']} → {metrics['after']['rows']}")
                print(f"  Null Values: {metrics['before']['null_values']} → {metrics['after']['null_values']}")
                
                # Show specific metrics if available
                if 'invalid_json' in metrics['before'] or 'invalid_json' in metrics['after']:
                    before = metrics['before'].get('invalid_json', 'N/A')
                    after = metrics['after'].get('invalid_json', 'N/A')
                    print(f"  Invalid JSON: {before} → {after}")
                    
                if 'invalid_task_ids' in metrics['before'] or 'invalid_task_ids' in metrics['after']:
                    before = metrics['before'].get('invalid_task_ids', 'N/A')
                    after = metrics['after'].get('invalid_task_ids', 'N/A')
                    print(f"  Invalid Task IDs: {before} → {after}")
    else:
        print("No issues to fix.")
else:
    print(f"Error: {analysis.get('message', 'Unknown error')}")
