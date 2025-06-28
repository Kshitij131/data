"""
Test script for GPT Agent's file analysis capabilities.
"""
import os
import json
from dotenv import load_dotenv
from gpt_agent import GPTAgent

def main():
    # Load environment variables
    load_dotenv(override=True)
    
    # Initialize the GPT agent
    print("Initializing GPT agent...")
    agent = GPTAgent()
    
    # Test file analysis
    print("\n=== Testing File Analysis ===")
    
    # CSV file analysis
    csv_file = "Clients_Sample_Inconsistent.csv"
    print(f"Analyzing CSV file: {csv_file}")
    
    if os.path.exists(csv_file):
        # Analyze the file
        analysis = agent.analyze_file(csv_file)
        
        # Print the summary
        print(f"\nAnalysis Status: {analysis['status']}")
        if analysis['status'] == 'success':
            print(f"File: {analysis['file_info']['file_name']}")
            print(f"Type: {analysis['file_info']['file_type']}")
            print(f"Rows: {analysis['file_info']['rows']}")
            print(f"Columns: {analysis['file_info']['columns']}")
            
            # Print summary information
            print(f"\nFound {analysis['summary']['total_issues']} issues:")
            print(f"- Missing values: {analysis['summary']['missing_values']}")
            print(f"- Duplicate rows: {analysis['summary']['duplicate_rows']}")
            print(f"- Column issues: {analysis['summary']['column_issues']}")
            
            # Print raw response if needed for debugging
            if analysis['summary']['total_issues'] == 0:
                print("\nNo issues found by LLM analysis. This could be due to:")
                print("1. Authentication error with the LLM")
                print("2. Rate limiting")
                print("3. The LLM not detecting issues")
                
                # Let's try our own analysis to find obvious issues
                print("\nPerforming manual analysis...")
                
                # Use pandas to analyze the file
                import pandas as pd
                df = pd.read_csv(csv_file)
                
                # Check for duplicate ClientIDs
                dup_clients = df['ClientID'].duplicated().sum()
                if dup_clients > 0:
                    print(f"Found {dup_clients} duplicate ClientIDs")
                    print("Duplicate ClientIDs:", df[df['ClientID'].duplicated(keep=False)]['ClientID'].tolist())
                
                # Check PriorityLevel range
                invalid_priority = df[~df['PriorityLevel'].between(1, 5)].shape[0]
                if invalid_priority > 0:
                    print(f"Found {invalid_priority} rows with PriorityLevel outside valid range (1-5)")
                    print("Invalid PriorityLevels:", df[~df['PriorityLevel'].between(1, 5)]['PriorityLevel'].tolist())
                
                # Check AttributesJSON format
                def is_valid_json(text):
                    try:
                        if isinstance(text, str):
                            json.loads(text)
                            return True
                    except:
                        return False
                    return False
                
                invalid_json = sum(~df['AttributesJSON'].apply(is_valid_json))
                if invalid_json > 0:
                    print(f"Found {invalid_json} rows with invalid JSON in AttributesJSON")
                
                # Check for invalid TaskIDs
                def has_invalid_task_id(task_list):
                    if not isinstance(task_list, str):
                        return False
                    tasks = task_list.split(',')
                    for task in tasks:
                        if not (task.startswith('T') and task[1:].isdigit()):
                            return True
                        elif task.startswith('T') and task[1:].isdigit() and int(task[1:]) > 50:
                            return True
                    return False
                
                invalid_tasks = sum(df['RequestedTaskIDs'].apply(has_invalid_task_id))
                if invalid_tasks > 0:
                    print(f"Found {invalid_tasks} rows with invalid TaskIDs in RequestedTaskIDs")
            
            # Print detailed issues if found
            if 'detailed_analysis' in analysis and 'issues' in analysis['detailed_analysis']:
                print("\nTop issues:")
                for i, issue in enumerate(analysis['detailed_analysis']['issues'][:5]):
                    print(f"{i+1}. {issue.get('type', 'Unknown')} - {issue.get('description', 'No description')}")
                    print(f"   Location: {issue.get('location', 'Unknown')}")
                    print(f"   Fix: {issue.get('fix', 'No fix recommendation')}")
                    print()
            
            # Test fixing the file
            print("\n=== Testing File Fix ===")
            fix_result = agent.fix_file_errors(csv_file)
            
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
                    
                    # Display before/after examples if available
                    if 'before_after_examples' in fix:
                        print("\n  Examples of changes:")
                        for example in fix['before_after_examples'][:3]:  # Show up to 3 examples
                            print(f"    Before: {example['before']}")
                            print(f"    After:  {example['after']}")
                            print()
                
                # Show failed fixes if any
                if fix_result['fixes_failed']:
                    print(f"\nFailed to apply {len(fix_result['fixes_failed'])} fixes:")
                    for i, fail in enumerate(fix_result['fixes_failed']):
                        print(f"  {i+1}. {fail.get('issue', {}).get('type', 'Unknown')} - {fail.get('reason', 'Unknown reason')}")
                
                # Compare before and after file statistics if desired
                try:
                    import pandas as pd
                    original_df = pd.read_csv(csv_file)
                    fixed_df = pd.read_csv(fix_result['fixed_file'])
                    
                    print("\nBefore vs After Statistics:")
                    print(f"  Rows: {len(original_df)} → {len(fixed_df)}")
                    
                    # Check if duplicate ClientIDs were fixed
                    orig_dup = original_df['ClientID'].duplicated().sum()
                    fixed_dup = fixed_df['ClientID'].duplicated().sum()
                    if orig_dup != fixed_dup:
                        print(f"  Duplicate ClientIDs: {orig_dup} → {fixed_dup}")
                    
                    # Check if PriorityLevel values were fixed
                    orig_invalid = original_df[~original_df['PriorityLevel'].between(1, 5)].shape[0]
                    fixed_invalid = fixed_df[~fixed_df['PriorityLevel'].between(1, 5)].shape[0]
                    if orig_invalid != fixed_invalid:
                        print(f"  Invalid PriorityLevels: {orig_invalid} → {fixed_invalid}")
                    
                    # Check for invalid JSON fixes
                    def is_valid_json(text):
                        try:
                            if isinstance(text, str):
                                json.loads(text)
                                return True
                        except:
                            return False
                        return False
                    
                    orig_invalid_json = sum(~original_df['AttributesJSON'].apply(is_valid_json))
                    fixed_invalid_json = sum(~fixed_df['AttributesJSON'].apply(is_valid_json))
                    if orig_invalid_json != fixed_invalid_json:
                        print(f"  Invalid JSON entries: {orig_invalid_json} → {fixed_invalid_json}")
                    
                    # Check for fixed invalid TaskIDs
                    def count_task_ids(task_list):
                        if not isinstance(task_list, str):
                            return 0
                        return len(task_list.split(','))
                    
                    orig_task_count = original_df['RequestedTaskIDs'].apply(count_task_ids).sum()
                    fixed_task_count = fixed_df['RequestedTaskIDs'].apply(count_task_ids).sum()
                    if orig_task_count != fixed_task_count:
                        print(f"  Total TaskIDs: {orig_task_count} → {fixed_task_count} (Invalid ones removed)")
                
                except Exception as e:
                    print(f"Error comparing files: {str(e)}")
            
        else:
            print(f"Error: {analysis['message']}")
    else:
        print(f"Error: File {csv_file} not found")

if __name__ == "__main__":
    main()
