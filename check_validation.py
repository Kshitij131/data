import pandas as pd
import json
from local_validator imp# Invalid task IDs
invalid_tasks = [e for e in errors if e.get('type') == 'invalid_task_ids']
if invalid_tasks:
    print(f"- Invalid task IDs: {len(invalid_tasks)}")
    for e in invalid_tasks[:5]:
        print(f"  * {e.get('message')} - Client: {e.get('details', {}).get('client_id', 'Unknown')}")
    
    # Check for T99 detection
    t99_errors = [e for e in invalid_tasks if 'T99' in str(e.get('details', {}).get('invalid_ids', []))]
    if t99_errors:
        print(f"  * Detected {len(t99_errors)} clients with T99 out-of-range task IDs")
    else:
        print("  * Warning: T99 out-of-range task IDs not detected")calValidator

# Read the CSV file
df = pd.read_csv("Clients_Sample_Inconsistent.csv")
raw_data = df.to_dict(orient="records")

# Initialize local validator
validator = LocalValidator()

# Before validation, print required columns
print(f"Before validation, required columns: {validator.required_columns}")

# Force client validation schema
validator.required_columns = validator.client_required_columns
print(f"Forcing client schema: {validator.required_columns}")

# Validate the data
errors = validator.validate_data(raw_data)

# After validation, print required columns to see if they changed
print(f"After validation, required columns: {validator.required_columns}")

# Count errors by type
error_types = {}
for error in errors:
    error_type = error.get('type', 'unknown')
    error_types[error_type] = error_types.get(error_type, 0) + 1

print("\nFound validation issues by type:")
for error_type, count in error_types.items():
    print(f"{error_type}: {count}")

# Print some key issues
print("\nKey issues found:")

# Duplicate IDs
dup_ids = [e for e in errors if e.get('type') == 'duplicate_id']
if dup_ids:
    print(f"- Duplicate IDs: {len(dup_ids)}")
    for e in dup_ids[:3]:
        print(f"  * {e.get('message')}")

# Invalid JSON
invalid_json = [e for e in errors if e.get('type') == 'invalid_json']
if invalid_json:
    print(f"- Invalid JSON: {len(invalid_json)}")
    for e in invalid_json[:3]:
        print(f"  * {e.get('message')}")

# Out of range values
out_of_range = [e for e in errors if e.get('type') == 'out_of_range']
if out_of_range:
    print(f"- Out of range values: {len(out_of_range)}")
    for e in out_of_range[:3]:
        print(f"  * {e.get('message')} - Value: {e.get('details', {}).get('value')}")

# Invalid task IDs
invalid_tasks = [e for e in errors if e.get('type') == 'invalid_task_ids']
if invalid_tasks:
    print(f"- Invalid task IDs: {len(invalid_tasks)}")
    for e in invalid_tasks[:3]:
        print(f"  * {e.get('message')} - Client: {e.get('details', {}).get('client_id')}")
