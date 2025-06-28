import sys
sys.path.append('.')
from gpt_agent import GPTAgent
import pandas as pd

def test_single_fix():
    """Test a single fix to verify the fix_file_errors function works correctly"""
    print("Testing fix_file_errors function...")
    
    # Create agent
    agent = GPTAgent()
    
    # Create a sample DataFrame with a specific issue to fix
    df = pd.DataFrame({
        'ClientID': ['C1', 'C2', 'C1', 'C4'],  # Duplicate ClientID
        'PriorityLevel': [3, 7, 2, 4],  # Out of range priority (7)
        'AttributesJSON': ['{"name": "test"}', 'invalid json', '{"id": 123}', 'another invalid'],
        'RequestedTaskIDs': ['T1,T2', 'TX,T50,T51', 'T3', 'T60,ABC']
    })
    
    # Save to CSV
    test_file = 'test_fix_sample.csv'
    df.to_csv(test_file, index=False)
    
    # Try fixing one issue at a time
    issue_types = ['duplicate_id', 'out_of_range', 'invalid_json', 'invalid_task_ids']
    
    for issue_type in issue_types:
        print(f"\nTesting fix for {issue_type}...")
        # Analyze the file first to get issues
        analysis = agent.analyze_file(test_file)
        
        # Find the issue ID for this type
        issue_id = None
        for i, issue in enumerate(analysis['detailed_analysis']['issues']):
            if issue.get('type') == issue_type:
                issue_id = i
                break
        
        if issue_id is not None:
            print(f"Found issue of type {issue_type} at index {issue_id}")
            # Try to fix just this issue
            result = agent.fix_file_errors(test_file, fix_ids=[issue_id], 
                                          output_path=f'fixed_{issue_type}.csv')
            print(f"Fix result: {result['status']}")
            print(f"Message: {result.get('message', '')}")
            
            if 'fixes_applied' in result:
                print(f"Applied {len(result['fixes_applied'])} fixes")
            
            if 'fixes_failed' in result and result['fixes_failed']:
                print("Failed fixes:")
                for fail in result['fixes_failed']:
                    print(f"  - {fail.get('reason', 'Unknown reason')}")
        else:
            print(f"No issue of type {issue_type} found")

if __name__ == "__main__":
    test_single_fix()
