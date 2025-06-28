import sys
sys.path.append('.')
from gpt_agent import GPTAgent
import json

agent = GPTAgent()
analysis = agent.analyze_file('Clients_Sample_Inconsistent.csv')

for issue in analysis['detailed_analysis']['issues']:
    print(f"Issue type: {issue.get('type')}")
    print(f"Fix code: {issue.get('fix_code', 'None')}")
    print("=" * 50)
