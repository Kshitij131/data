import pandas as pd
import json
from gemini import GeminiAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_priority_validation():
    print("Testing PriorityLevel validation...")
    
    # Initialize the Gemini agent
    try:
        gemini = GeminiAgent()
        print(f"Successfully initialized Gemini with model: {gemini.model.model_name}")
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        return
    
    # Create test data with different priority levels
    test_data = [
        {"ClientID": "C1", "ClientName": "Test Client 1", "PriorityLevel": 1, "GroupTag": "Group1"},
        {"ClientID": "C2", "ClientName": "Test Client 2", "PriorityLevel": 3, "GroupTag": "Group1"},
        {"ClientID": "C3", "ClientName": "Test Client 3", "PriorityLevel": 5, "GroupTag": "Group1"},
        {"ClientID": "C4", "ClientName": "Test Client 4", "PriorityLevel": 7, "GroupTag": "Group1"},
        {"ClientID": "C5", "ClientName": "Test Client 5", "PriorityLevel": 0, "GroupTag": "Group1"},
    ]
    
    print("\nTest data:")
    for client in test_data:
        print(f"Client {client['ClientID']} - PriorityLevel: {client['PriorityLevel']}")
    
    # Convert to JSON for Gemini
    data_json = json.dumps(test_data)
    
    # Send to Gemini for validation
    print("\nSending data to Gemini for validation...")
    try:
        validation_result = gemini.validate_data(data_json)
        
        # Process and display validation results
        errors = validation_result.get("errors", [])
        if errors:
            print(f"\nFound {len(errors)} validation issues:")
            
            # Filter for priority level errors
            priority_errors = [e for e in errors if "priority" in e.get("message", "").lower()]
            if priority_errors:
                print("\nPriority level errors:")
                for i, error in enumerate(priority_errors, 1):
                    print(f"  {i}. {error.get('message', 'No message')}")
                    if "details" in error:
                        print(f"     Details: {json.dumps(error.get('details'))}")
            else:
                print("\nNo priority level errors found.")
        else:
            print("\nNo validation issues found!")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    test_priority_validation()
