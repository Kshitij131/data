import sys
sys.path.append('.')
from gpt_agent import GPTAgent
import pandas as pd

agent = GPTAgent()
analysis = agent.analyze_file('Clients_Sample_Inconsistent.csv')

# Check issue IDs
print("Issue IDs and types:")
for i, issue in enumerate(analysis['detailed_analysis']['issues']):
    print(f"{i}: {issue.get('type')} - {issue.get('location')}")

# Try each fix individually and print results
for i, issue in enumerate(analysis['detailed_analysis']['issues']):
    print(f"\nTesting fix for issue {i}: {issue.get('type')} - {issue.get('location')}")
    fix_result = agent.fix_file_errors('Clients_Sample_Inconsistent.csv', fix_ids=[i])
    print(f"Status: {fix_result['status']}")
    print(f"Message: {fix_result.get('message', 'No message')}")
    if 'fixes_applied' in fix_result:
        print(f"Fixes applied: {len(fix_result['fixes_applied'])}")
    if 'fixes_failed' in fix_result:
        print(f"Fixes failed: {len(fix_result['fixes_failed'])}")
        for fail in fix_result['fixes_failed']:
            print(f"  Reason: {fail.get('reason', 'Unknown')}")

# Now try with explicit fix_ids
print("\nTrying with explicit fix_ids=[0, 1, 2, 3]")
fix_result = agent.fix_file_errors('Clients_Sample_Inconsistent.csv', fix_ids=[0, 1, 2, 3])
print(f"Status: {fix_result['status']}")
print(f"Message: {fix_result.get('message', 'No message')}")
if 'fixes_applied' in fix_result:
    print(f"Fixes applied: {len(fix_result['fixes_applied'])}")
if 'fixes_failed' in fix_result:
    print(f"Fixes failed: {len(fix_result['fixes_failed'])}")
    for fail in fix_result['fixes_failed']:
        print(f"  Reason: {fail.get('reason', 'Unknown')}")
