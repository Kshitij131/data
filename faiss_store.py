import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
from typing import List, Dict, Any
import re

class ValidationError:
    def __init__(self, error_type: str, message: str, details: Dict[str, Any] = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}

class FAISSSearcher:
    def __init__(self, mode="headers"):
        self.mode = mode
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.entries = []
        self.required_columns = [
            "ClientID", "WorkerID", "TaskID", "PriorityLevel", 
            "Duration", "AvailableSlots", "RequiredSkills",
            "MaxConcurrent", "Phase", "AttributesJSON"
        ]

        if mode == "headers":
            file = os.path.join("data", "correct_headers.json")
            with open(file) as f:
                self.entries = json.load(f)
        elif mode == "rows":
            file = os.path.join("data", "sample_rows.json")
            with open(file) as f:
                data = json.load(f)
                # Store both the JSON object and its string representation
                self.entries_obj = data
                self.entries = [json.dumps(row) for row in data]

        self.embeddings = self.model.encode(self.entries)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))

    def search(self, query: str, top_k=1):
        """Search for similar entries in the FAISS index"""
        query_vec = self.model.encode([query])
        D, I = self.index.search(np.array(query_vec), top_k)
        results = [self.entries[i] for i in I[0]]
        
        # If searching rows, parse JSON strings back to objects
        if self.mode == "rows":
            results = [json.loads(r) if isinstance(r, str) else r for r in results]
        
        return results

    def find_similar_errors(self, error_message: str, top_k=3):
        """Find similar validation errors from known patterns"""
        query_vec = self.model.encode([error_message])
        D, I = self.index.search(np.array(query_vec), top_k)
        return [self.entries[i] for i in I[0]]
    
    def validate_data(self, data: List[Dict[str, Any]]) -> List[ValidationError]:
        errors = []
        
        # Store all IDs for duplicate checking
        client_ids = set()
        worker_ids = set()
        task_ids = set()
        
        # Store task durations per phase for saturation check
        phase_durations = {}
        
        # Store worker skills for coverage check
        worker_skills = {}
        required_skills = set()
        
        # Store co-run groups for circular dependency check
        corun_groups = {}
        
        for row in data:
            # a. Check missing columns
            missing_cols = [col for col in self.required_columns if col not in row]
            if missing_cols:
                errors.append(ValidationError("missing_columns", 
                    f"Missing required columns: {', '.join(missing_cols)}", 
                    {"row": row}))
                continue  # Skip further validation if missing columns
            
            # b. Check duplicate IDs
            if row["ClientID"] in client_ids:
                errors.append(ValidationError("duplicate_id", 
                    f"Duplicate ClientID: {row['ClientID']}", 
                    {"id": row["ClientID"]}))
            client_ids.add(row["ClientID"])
            
            if row["WorkerID"] in worker_ids:
                errors.append(ValidationError("duplicate_id", 
                    f"Duplicate WorkerID: {row['WorkerID']}", 
                    {"id": row["WorkerID"]}))
            worker_ids.add(row["WorkerID"])
            
            if row["TaskID"] in task_ids:
                errors.append(ValidationError("duplicate_id", 
                    f"Duplicate TaskID: {row['TaskID']}", 
                    {"id": row["TaskID"]}))
            task_ids.add(row["TaskID"])
            
            # c. Check malformed lists
            try:
                slots = json.loads(row["AvailableSlots"]) if isinstance(row["AvailableSlots"], str) else row["AvailableSlots"]
                if not all(isinstance(x, (int, float)) for x in slots):
                    errors.append(ValidationError("malformed_list", 
                        f"AvailableSlots contains non-numeric values", 
                        {"row": row}))
            except (json.JSONDecodeError, TypeError):
                errors.append(ValidationError("malformed_list", 
                    "Invalid AvailableSlots format", 
                    {"row": row}))
            
            # d. Check out-of-range values
            if not (1 <= row["PriorityLevel"] <= 5):
                errors.append(ValidationError("out_of_range", 
                    f"PriorityLevel must be between 1 and 5", 
                    {"value": row["PriorityLevel"]}))
            
            if row["Duration"] < 1:
                errors.append(ValidationError("out_of_range", 
                    "Duration must be at least 1", 
                    {"value": row["Duration"]}))
            
            # e. Check broken JSON
            try:
                # Handle AttributesJSON properly whether it's a string or object
                if isinstance(row["AttributesJSON"], str):
                    attrs = json.loads(row["AttributesJSON"])
                else:
                    attrs = row["AttributesJSON"]
                
                if not isinstance(attrs, dict):
                    errors.append(ValidationError("invalid_json", 
                        "AttributesJSON must be a valid JSON object", 
                        {"row": row}))
            except (json.JSONDecodeError, TypeError):
                errors.append(ValidationError("invalid_json", 
                    "Invalid AttributesJSON format", 
                    {"row": row}))
            
            # Track data for complex validations
            phase = row["Phase"]
            if phase not in phase_durations:
                phase_durations[phase] = 0
            phase_durations[phase] += row["Duration"]
            
            # Track worker skills
            if isinstance(row.get("WorkerSkills"), list):
                worker_skills[row["WorkerID"]] = set(row["WorkerSkills"])
            
            # Track required skills
            if isinstance(row.get("RequiredSkills"), list):
                required_skills.update(row["RequiredSkills"])
            
            # Track co-run groups
            if "CoRunGroup" in row:
                corun_groups[row["TaskID"]] = row["CoRunGroup"]
        
        # f. Check unknown references
        all_task_ids = set(task_ids)
        for row in data:
            if "RequestedTaskIDs" in row:
                unknown_tasks = set(row["RequestedTaskIDs"]) - all_task_ids
                if unknown_tasks:
                    errors.append(ValidationError("unknown_reference", 
                        f"Referenced unknown tasks: {unknown_tasks}", 
                        {"task": row["TaskID"]}))
        
        # g. Check circular co-run groups
        def find_cycle(graph, start, visited=None, path=None):
            if visited is None:
                visited = set()
            if path is None:
                path = []
            
            visited.add(start)
            path.append(start)
            
            for next_node in graph.get(start, []):
                if next_node not in visited:
                    if find_cycle(graph, next_node, visited, path):
                        return True
                elif next_node in path:
                    return True
            
            path.pop()
            return False
        
        for task_id in corun_groups:
            if find_cycle(corun_groups, task_id):
                errors.append(ValidationError("circular_dependency", 
                    "Detected circular dependency in co-run groups", 
                    {"start_task": task_id}))
        
        # h. Check phase-window constraints
        # This would require additional context about phase windows
        
        # i. Check worker load
        for row in data:
            if len(row.get("AvailableSlots", [])) < row.get("MaxLoadPerPhase", 0):
                errors.append(ValidationError("overloaded_worker", 
                    f"Worker {row['WorkerID']} has insufficient slots", 
                    {"worker": row["WorkerID"]}))
        
        # j. Check phase-slot saturation
        for phase, total_duration in phase_durations.items():
            total_slots = sum(len(row.get("AvailableSlots", [])) 
                            for row in data if row["Phase"] == phase)
            if total_duration > total_slots:
                errors.append(ValidationError("phase_saturation", 
                    f"Phase {phase} is oversaturated", 
                    {"phase": phase, "duration": total_duration, "slots": total_slots}))
        
        # k. Check skill coverage
        for skill in required_skills:
            skilled_workers = sum(1 for skills in worker_skills.values() if skill in skills)
            if skilled_workers == 0:
                errors.append(ValidationError("skill_coverage", 
                    f"No workers available for skill: {skill}", 
                    {"skill": skill}))
        
        # l. Check max-concurrency feasibility
        for row in data:
            qualified_workers = sum(1 for w_skills in worker_skills.values() 
                                if all(skill in w_skills for skill in row.get("RequiredSkills", [])))
            if row.get("MaxConcurrent", 0) > qualified_workers:
                errors.append(ValidationError("concurrency_infeasible", 
                    f"MaxConcurrent exceeds available qualified workers", 
                    {"task": row["TaskID"], "required": row["MaxConcurrent"], "available": qualified_workers}))
        
        return errors
