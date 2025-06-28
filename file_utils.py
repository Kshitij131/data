import pandas as pd
import json
from typing import Dict, List, Any, Union
import io

def process_excel_file(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Process an Excel file and convert it to a list of dictionaries
    """
    # Read the Excel file into a pandas DataFrame
    try:
        # Try first as xlsx
        df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
    except Exception:
        try:
            # Try as xls if xlsx fails
            df = pd.read_excel(io.BytesIO(file_bytes), engine='xlrd')
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {str(e)}")
    
    # Convert DataFrame to list of dictionaries
    records = df.to_dict(orient='records')
    
    # Clean up the data
    clean_records = []
    for record in records:
        # Remove NaN values
        clean_record = {}
        for key, value in record.items():
            if pd.isna(value):
                # Skip NaN values or replace with None
                clean_record[key] = None
            elif isinstance(value, (int, float)):
                # Keep numeric values as is
                clean_record[key] = value
            else:
                # Convert all other values to strings
                clean_record[key] = str(value)
        
        clean_records.append(clean_record)
    
    return clean_records

def convert_to_json(data: List[Dict[str, Any]]) -> str:
    """
    Convert a list of dictionaries to a JSON string
    """
    return json.dumps(data, ensure_ascii=False)

def parse_json_safely(json_str: str) -> Union[Dict, List]:
    """
    Parse a JSON string safely, handling different formats and common errors
    """
    try:
        # Try to parse as-is first
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to find valid JSON object/array in the string
        try:
            # Look for JSON object
            obj_start = json_str.find('{')
            obj_end = json_str.rfind('}') + 1
            if obj_start >= 0 and obj_end > obj_start:
                return json.loads(json_str[obj_start:obj_end])
            
            # Look for JSON array
            arr_start = json_str.find('[')
            arr_end = json_str.rfind(']') + 1
            if arr_start >= 0 and arr_end > arr_start:
                return json.loads(json_str[arr_start:arr_end])
        except:
            pass
    
    # If all parsing fails, return empty dict
    return {}
    """
    Parse a JSON string safely, handling common errors
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
