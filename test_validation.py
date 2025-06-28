import pandas as pd
import json
from file_utils import process_excel_file, convert_to_json
from local_validator import LocalValidator
import sys

def test_csv_validation():
    """Test local validation on the inconsistent CSV file"""
    print("Testing CSV validation on Clients_Sample_Inconsistent.csv")
    
    # Read the CSV file
    df = pd.read_csv("Clients_Sample_Inconsistent.csv")
    raw_data = df.to_dict(orient="records")
    
    # Print sample data
    print(f"\nSample data (first 2 rows):")
    for i, row in enumerate(raw_data[:2]):
        print(f"Row {i+1}: {row}")
    
    # Initialize local validator
    validator = LocalValidator()
    
    # Validate the data
    print("\nValidating data...")
    errors = validator.validate_data(raw_data)
    
    # Group errors by type for clearer reporting
    error_types = {}
    for error in errors:
        error_type = error.get('type', 'unknown')
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error)
    
    # Print errors by type
    if errors:
        print(f"\nFound {len(errors)} validation issues:")
        for error_type, type_errors in error_types.items():
            print(f"\n{error_type.upper()} ({len(type_errors)} issues):")
            for i, error in enumerate(type_errors[:5]):  # Show at most 5 per type
                detail = ""
                if "client_id" in error.get("details", {}):
                    detail = f" - Client: {error['details']['client_id']}"
                elif "id" in error.get("details", {}):
                    detail = f" - {error['details']['id']}"
                print(f"  {i+1}. {error.get('message', 'No message')}{detail}")
            if len(type_errors) > 5:
                print(f"  ... and {len(type_errors) - 5} more {error_type} issues")
    else:
        print("\nNo validation errors found!")

    # Test specific problematic rows 
    print("\nChecking specific problematic rows:")
    problem_clients = ["C1", "C5", "C20", "C21", "C25", "C33", "C42", "C45", "C50"]
    for client_id in problem_clients:
        client_rows = [row for row in raw_data if row.get("ClientID") == client_id]
        for row in client_rows:
            issues = []
            # Check for duplicate ClientID
            if client_id == "C1":
                issues.append("Duplicate ClientID")
            
            # Check PriorityLevel out of range
            if "PriorityLevel" in row and row["PriorityLevel"] > 5:
                issues.append(f"PriorityLevel out of range: {row['PriorityLevel']}")
            
            # Check invalid TaskIDs
            if "RequestedTaskIDs" in row:
                task_ids = row["RequestedTaskIDs"]
                if "TX" in task_ids:
                    issues.append("Contains invalid TaskID: TX")
                if "T99" in task_ids:
                    issues.append("Contains out-of-range TaskID: T99")
                if "T60" in task_ids:
                    issues.append("Contains out-of-range TaskID: T60")
            
            # Check invalid JSON
            if "AttributesJSON" in row:
                try:
                    if isinstance(row["AttributesJSON"], str) and not row["AttributesJSON"].startswith("{"):
                        issues.append("Invalid JSON format in AttributesJSON")
                except:
                    issues.append("Error parsing AttributesJSON")
            
            if issues:
                print(f"Client {client_id}: {', '.join(issues)}")

if __name__ == "__main__":
    test_csv_validation()
