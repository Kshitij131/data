from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from dotenv import load_dotenv
from gpt_agent import GPTAgent
from local_validator import LocalValidator
from io import BytesIO
import json
import pandas as pd
from file_utils import process_excel_file, convert_to_json
from typing import List, Dict, Any, Optional

load_dotenv()

app = FastAPI(
    title="Data Alchemist AI API",
    description="AI-powered spreadsheet validation, search, rule generation & correction using OpenAI GPT.",
    version="1.0.0"
)

# Initialize GPT Agent and Local Validator
try:
    gpt_agent = GPTAgent()
    gpt_available = True
    print("✅ GPT Agent initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Failed to initialize GPT: {e}")
    print("Continuing with local validator only")
    gpt_agent = None
    gpt_available = False

# Always initialize the local validator as a fallback
local_validator = LocalValidator()

# Base input model for all text-based endpoints
class Query(BaseModel):
    text: str
    top_k: int = 1  # Optional: unused for now

# Root route
@app.get("/")
def root():
    return {"message": "Welcome to the Data Alchemist AI API 🚀"}

# 1. AI Header Mapper
@app.post("/ai-header-map")
def map_header(query: Query):
    result = gpt_agent.map_header(query.text)
    return {"mapped_header": result}

# 2. AI Validator (row check)
@app.post("/ai-validate")
def validate_row(query: Query):
    if gpt_available:
        # Try using GPT first
        try:
            result = gpt_agent.validate_data(query.text)
            return {"issues": result, "using_gpt": True}
        except Exception as e:
            print(f"GPT validation failed: {e}")
            # Fall back to local validation only as a last resort
    
    # Local validation fallback - only used if GPT is unavailable or fails
    try:
        data = json.loads(query.text) if isinstance(query.text, str) else query.text
        if isinstance(data, dict):
            data = [data]
        errors = local_validator.validate_data(data)
        return {"issues": {"errors": errors}, "using_gpt": False}
    except Exception as e:
        return {"issues": {"errors": [{"type": "error", "message": f"Validation error: {str(e)}"}]}, "using_gpt": False}

# 3. Natural Language Search → JSON Filter
@app.post("/ai-nl-search")
def nl_search_filter(query: Query):
    result = gpt_agent.query_to_filter(query.text)
    return {"filter": result}

# 4. Natural Language → Data Modification Command
@app.post("/ai-nl-modify")
def modify_data(query: Query):
    result = gpt_agent.modify_data(query.text)
    return {"modification": result}

# 5. Auto Suggest Data Correction
@app.post("/ai-correct")
def suggest_fix(query: Query):
    if gpt_available:
        try:
            result = gpt_agent.suggest_correction(query.text)
            return {"suggested_fix": result, "using_gpt": True}
        except Exception as e:
            print(f"GPT correction failed: {e}")
            # Fall back to local correction only if GPT fails
    
    # Local fallback for suggestions - only used if GPT is unavailable
    try:
        data = json.loads(query.text) if isinstance(query.text, str) else query.text
        if isinstance(data, dict):
            data = [data]
            
        # Apply simple fixes where possible
        fixed_data = data.copy()
        for i, row in enumerate(fixed_data):
            # Fix priority level if out of range
            if "PriorityLevel" in row:
                try:
                    priority = int(row["PriorityLevel"])
                    if priority < 1:
                        row["PriorityLevel"] = 1
                    elif priority > 5:
                        row["PriorityLevel"] = 5
                except (ValueError, TypeError):
                    row["PriorityLevel"] = 3  # Default middle value
                    
            # Fix duration if less than 1
            if "Duration" in row and (not isinstance(row["Duration"], (int, float)) or row["Duration"] < 1):
                row["Duration"] = 1
                
            # Convert AttributesJSON to proper format
            if "AttributesJSON" in row and isinstance(row["AttributesJSON"], str):
                try:
                    row["AttributesJSON"] = json.loads(row["AttributesJSON"])
                except json.JSONDecodeError:
                    row["AttributesJSON"] = {}
            
        return {
            "suggested_fix": {
                "corrected_data": fixed_data,
                "corrections": [{"message": "Basic corrections applied"}],
                "confidence_score": 0.7,
                "using_local_correction": True
            },
            "using_gpt": False
        }
    except Exception as e:
        return {"suggested_fix": {"error": str(e)}, "using_gpt": False}

# 6. NL → Rule JSON Generator
@app.post("/ai-rule-gen")
def generate_rule(query: Query):
    result = gpt_agent.nl_to_rule(query.text)
    return {"rule": result}

# 7. Rule Recommendations (from data context)
@app.post("/ai-rule-hints")
def rule_hints(query: Query):
    result = gpt_agent.rule_recommendations(query.text)
    return {"recommendations": result}

# 8. Priority Profile Suggestion
@app.post("/ai-priority")
def priority_weights(query: Query):
    result = gpt_agent.generate_priority_profile(query.text)
    return {"weights": result}

# 9. File Upload: CSV or XLSX file
@app.post("/upload-data/")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    content = await file.read()

    try:
        # Process the file based on extension
        if ext == "csv":
            df = pd.read_csv(BytesIO(content))
            raw_data = df.to_dict(orient="records")
        elif ext in ["xlsx", "xls"]:
            # Use our utility function to process Excel properly
            raw_data = process_excel_file(content)
        else:
            return {"error": "❌ Unsupported file format. Please upload CSV or Excel."}

        # Convert data to proper JSON
        data_json = convert_to_json(raw_data)
        
        # Preview first 5 rows
        preview = raw_data[:5] if len(raw_data) > 5 else raw_data
        
        # Validate the data - try GPT first, then fall back to local validation
        if gpt_available:
            try:
                # Always use GPT for validation when available
                validation_result = gpt_agent.validate_data(data_json)
                return {
                    "message": "✅ File uploaded and processed successfully!",
                    "preview": preview,
                    "validation": validation_result,
                    "using_gpt": True,
                    "using_local_validation": False
                }
            except Exception as e:
                print(f"GPT validation failed: {e}")
                # Fall back to local validation only if GPT fails completely
                errors = local_validator.validate_data(raw_data)
                validation_result = {"errors": errors}
                return {
                    "message": "✅ File processed with basic validation (GPT unavailable)!",
                    "preview": preview,
                    "validation": validation_result,
                    "using_gpt": False,
                    "using_local_validation": True
                }
        else:
            # Use local validation if GPT is not available
            errors = local_validator.validate_data(raw_data)
            validation_result = {"errors": errors}
            return {
                "message": "✅ File processed with basic validation (GPT unavailable)!",
                "preview": preview,
                "validation": validation_result,
                "using_gpt": False,
                "using_local_validation": True
            }

    except Exception as e:
        return {"error": f"❌ Failed to process file: {str(e)}"}
