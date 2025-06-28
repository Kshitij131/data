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
import os
from fastapi.responses import FileResponse

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
    
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as f:
        f.write(content)

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
        
        # Use our comprehensive file analysis
        if gpt_available:
            try:
                # Use GPT agent's file analysis capability
                analysis_result = gpt_agent.analyze_file(temp_file_path)
                
                # Offer to fix the file if issues were found
                fix_info = None
                if analysis_result["status"] == "success" and analysis_result["summary"]["total_issues"] > 0:
                    # Prepare but don't execute fix yet - this will be done on demand from frontend
                    fix_info = {
                        "available": True,
                        "total_issues": analysis_result["summary"]["total_issues"],
                        "fixed_file_name": f"{file.filename.split('.')[0]}_fixed.{ext}"
                    }
                
                return {
                    "message": "✅ File uploaded and processed successfully!",
                    "preview": preview,
                    "analysis": analysis_result,
                    "fix_info": fix_info,
                    "temp_file_path": temp_file_path,
                    "using_gpt": True,
                    "using_local_validation": False
                }
            except Exception as e:
                print(f"Error using GPT analysis: {e}")
                # Fall back to basic validation
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

@app.post("/fix-file/")
async def fix_file(request: dict):
    """Fix issues in a previously uploaded file"""
    try:
        file_path = request.get("file_path")
        if not file_path or not os.path.exists(file_path):
            return {"error": "❌ Invalid file path or file not found"}
        
        # Use the GPT agent to fix the file
        fix_result = gpt_agent.fix_file_errors(file_path)
        
        if fix_result["status"] == "success" and fix_result["fixed_file"]:
            # Read the fixed file to return its content
            fixed_file_path = fix_result["fixed_file"]
            
            if fixed_file_path.endswith(".csv"):
                df_fixed = pd.read_csv(fixed_file_path)
            elif fixed_file_path.endswith((".xlsx", ".xls")):
                df_fixed = pd.read_excel(fixed_file_path)
            else:
                return {"error": "❌ Unknown file format after fixing"}
                
            fixed_preview = df_fixed.head(5).to_dict(orient="records")
            
            return {
                "message": f"✅ Successfully fixed {len(fix_result['fixes_applied'])} issues in the file",
                "fixed_file": os.path.basename(fixed_file_path),
                "fixed_preview": fixed_preview,
                "fixes_applied": fix_result["fixes_applied"],
                "fixes_failed": fix_result["fixes_failed"],
                "download_url": f"/download/{os.path.basename(fixed_file_path)}"
            }
        else:
            return {"error": "❌ Failed to fix the file", "details": fix_result}
            
    except Exception as e:
        return {"error": f"❌ Error fixing file: {str(e)}"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a fixed file"""
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path, 
        filename=filename,
        media_type="application/octet-stream"
    )
