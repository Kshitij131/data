import json
import re
from typing import Dict, List, Any

class LocalValidator:
    """
    A local validator class that can validate data without needing to call Gemini.
    This is a simple validation class that implements basic checks.
    """
    def __init__(self):
        self.required_columns = [
            "TaskID", "WorkerID", "ClientID", "Duration", 
            "PriorityLevel", "Phase"
        ]
    
    def validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        errors = []
        
        # Store all IDs for duplicate checking
        client_ids = set()
        worker_ids = set()
        task_ids = set()
        
        for row in data:
            row_errors = []
            
            # a. Check missing columns
            missing_cols = [col for col in self.required_columns if col not in row]
            if missing_cols:
                row_errors.append({
                    "type": "missing_columns",
                    "message": f"Missing required columns: {', '.join(missing_cols)}",
                    "details": {"row": row}
                })
            
            # b. Check duplicate IDs
            if "ClientID" in row:
                client_id = str(row["ClientID"])
                if client_id in client_ids:
                    row_errors.append({
                        "type": "duplicate_id",
                        "message": f"Duplicate ClientID: {client_id}",
                        "details": {"id": client_id}
                    })
                client_ids.add(client_id)
            
            if "WorkerID" in row:
                worker_id = str(row["WorkerID"])
                if worker_id in worker_ids:
                    row_errors.append({
                        "type": "duplicate_id",
                        "message": f"Duplicate WorkerID: {worker_id}",
                        "details": {"id": worker_id}
                    })
                worker_ids.add(worker_id)
            
            if "TaskID" in row:
                task_id = str(row["TaskID"])
                if task_id in task_ids:
                    row_errors.append({
                        "type": "duplicate_id",
                        "message": f"Duplicate TaskID: {task_id}",
                        "details": {"id": task_id}
                    })
                task_ids.add(task_id)
            
            # c. Check malformed lists
            if "AvailableSlots" in row:
                slots = row["AvailableSlots"]
                if isinstance(slots, str):
                    try:
                        slots = json.loads(slots)
                    except json.JSONDecodeError:
                        row_errors.append({
                            "type": "malformed_list",
                            "message": "Invalid AvailableSlots format",
                            "details": {"row": row}
                        })
                
                if isinstance(slots, list):
                    if not all(isinstance(x, (int, float)) for x in slots):
                        row_errors.append({
                            "type": "malformed_list",
                            "message": "AvailableSlots contains non-numeric values",
                            "details": {"row": row}
                        })
            
            # d. Check out-of-range values
            if "PriorityLevel" in row:
                try:
                    priority = int(row["PriorityLevel"])
                    if not (1 <= priority <= 5):
                        row_errors.append({
                            "type": "out_of_range",
                            "message": "PriorityLevel must be between 1 and 5",
                            "details": {"value": priority}
                        })
                except (ValueError, TypeError):
                    row_errors.append({
                        "type": "invalid_value",
                        "message": "PriorityLevel must be a number",
                        "details": {"value": row["PriorityLevel"]}
                    })
            
            if "Duration" in row:
                try:
                    duration = float(row["Duration"])
                    if duration < 1:
                        row_errors.append({
                            "type": "out_of_range",
                            "message": "Duration must be at least 1",
                            "details": {"value": duration}
                        })
                except (ValueError, TypeError):
                    row_errors.append({
                        "type": "invalid_value",
                        "message": "Duration must be a number",
                        "details": {"value": row["Duration"]}
                    })
            
            # e. Check JSON fields
            if "AttributesJSON" in row:
                attr_json = row["AttributesJSON"]
                if isinstance(attr_json, str):
                    try:
                        json.loads(attr_json)
                    except json.JSONDecodeError:
                        row_errors.append({
                            "type": "invalid_json",
                            "message": "Invalid AttributesJSON format",
                            "details": {"row": row}
                        })
                elif not isinstance(attr_json, dict):
                    row_errors.append({
                        "type": "invalid_json",
                        "message": "AttributesJSON must be a valid JSON object",
                        "details": {"row": row}
                    })
            
            # f. Check for invalid TaskIDs in RequestedTaskIDs
            if "RequestedTaskIDs" in row:
                task_ids_str = row["RequestedTaskIDs"]
                if isinstance(task_ids_str, str):
                    task_list = [tid.strip() for tid in task_ids_str.split(",")]
                    invalid_ids = [tid for tid in task_list if not (tid.startswith("T") and tid[1:].isdigit())]
                    
                    if invalid_ids:
                        row_errors.append({
                            "type": "invalid_task_ids",
                            "message": f"Invalid TaskIDs found: {', '.join(invalid_ids)}",
                            "details": {
                                "invalid_ids": invalid_ids,
                                "client_id": row.get("ClientID", "Unknown")
                            }
                        })
            
            # Add errors for this row
            errors.extend(row_errors)
        
        return errors
