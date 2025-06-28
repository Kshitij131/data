import os
import json
import pandas as pd
from dotenv import load_dotenv
from gemini import GeminiAgent

# Load environment variables
load_dotenv()

def test_gemini_validation():
    print("Testing Gemini validation on inconsistent CSV data...")
    
    # Initialize the Gemini agent
    try:
        gemini = GeminiAgent()
        print(f"Successfully initialized Gemini with model: {gemini.model.model_name}")
    except Exception as e:
        print(f"Failed to initialize Gemini: {e}")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv("Clients_Sample_Inconsistent.csv")
        data = df.to_dict(orient="records")
        print(f"Successfully loaded CSV with {len(data)} rows")
        
        # Print first 2 rows for reference
        for i, row in enumerate(data[:2]):
            print(f"\nRow {i+1}:")
            for key, value in row.items():
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return
    
    # Convert to JSON for Gemini
    data_json = json.dumps(data)
    
    # Send to Gemini for validation
    print("\nSending data to Gemini for validation...")
    try:
        validation_result = gemini.validate_data(data_json)
        
        # Process and display validation results
        errors = validation_result.get("errors", [])
        if errors:
            print(f"\nFound {len(errors)} validation issues:")
            
            # Check for raw response
            raw_response = None
            for error in errors:
                if error.get("type") == "validation_summary" and "details" in error:
                    raw_response = error.get("details", {}).get("full_response")
                    break
            
            if raw_response:
                print("\nRaw Gemini Response (first 500 chars):")
                print(raw_response[:500])
                print("...")
            
            # Group errors by type
            error_types = {}
            for error in errors:
                error_type = error.get("type", "unknown")
                if error_type not in error_types:
                    error_types[error_type] = []
                error_types[error_type].append(error)
            
            # Display errors grouped by type
            for error_type, type_errors in error_types.items():
                print(f"\n{error_type.upper()} ({len(type_errors)} issues):")
                for i, error in enumerate(type_errors[:5], 1):  # Show max 5 per type
                    print(f"  {i}. {error.get('message', 'No message')}")
                if len(type_errors) > 5:
                    print(f"  ... and {len(type_errors) - 5} more {error_type} issues")
        else:
            print("\nNo validation issues found! This is unexpected for an inconsistent file.")
    except Exception as e:
        print(f"Validation failed: {e}")

if __name__ == "__main__":
    test_gemini_validation()
