import os
import json
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage, 
    UserMessage
)
from azure.core.credentials import AzureKeyCredential
from faiss_store import FAISSSearcher
from typing import Dict, List, Any, Union
import pandas as pd
import re
from pathlib import Path

# Available GitHub AI models
GITHUB_AI_MODELS = {
    # OpenAI models
    "GPT_4": "openai/gpt-4",
    "GPT_4_TURBO": "openai/gpt-4-turbo",
    "GPT_4_1": "openai/gpt-4.1",
    "GPT_3_5_TURBO": "openai/gpt-3.5-turbo",
    
    # Meta models
    "LLAMA_3_1_8B": "meta/Meta-Llama-3.1-8B-Instruct",
    "LLAMA_3_1_70B": "meta/Meta-Llama-3.1-70B-Instruct",
    "LLAMA_3_1_405B": "meta/Meta-Llama-3.1-405B-Instruct",
    
    # Anthropic models
    "CLAUDE_3_OPUS": "anthropic/claude-3-opus",
    "CLAUDE_3_SONNET": "anthropic/claude-3-sonnet",
    "CLAUDE_3_HAIKU": "anthropic/claude-3-haiku",
}

class GPTAgent:
    def __init__(self):
        # Initialize with GitHub AI
        github_token = os.getenv("GITHUB_TOKEN")
        
        if not github_token:
            raise ValueError("GitHub token not found. Please set GITHUB_TOKEN in your environment variables.")
            
        # Get GitHub AI endpoint and model from environment variables
        github_ai_endpoint = os.getenv("GITHUB_AI_ENDPOINT", "https://models.github.ai/inference")
        github_ai_model = os.getenv("GITHUB_AI_MODEL", "meta/Meta-Llama-3.1-405B-Instruct")
        
        # GitHub AI requires the token to have "models:read" permission
        # Make sure you have granted this permission to your token in GitHub settings
        print("⚠️ Note: Your GitHub token must have 'models:read' permission to access GitHub AI models")
        
        # Configure client with proper error handling and retry policies
        self.client = ChatCompletionsClient(
            endpoint=github_ai_endpoint,
            credential=AzureKeyCredential(github_token)
        )
        
        # Normalize the model name with proper prefixes to avoid bad requests
        if github_ai_model in GITHUB_AI_MODELS:
            # If the model name is a key in our dictionary, use the mapped value
            github_ai_model = GITHUB_AI_MODELS[github_ai_model]
        elif not any(github_ai_model.startswith(prefix) for prefix in ["openai/", "meta/", "meta-llama/", "anthropic/", "deepseek"]):
            # If no prefix is present, try to infer the correct prefix
            if "gpt" in github_ai_model.lower():
                github_ai_model = f"openai/{github_ai_model}"
            elif "llama" in github_ai_model.lower():
                github_ai_model = f"meta/{github_ai_model}"
            elif "claude" in github_ai_model.lower():
                github_ai_model = f"anthropic/{github_ai_model}"
            elif "deepseek" in github_ai_model.lower():
                github_ai_model = f"deepseek/{github_ai_model}"
        
        # Store the original model name for reference
        self.original_model_name = github_ai_model
        
        # Validate the model and use fallbacks if needed
        if self._validate_model(github_ai_model):
            self.model_name = github_ai_model
        else:
            print(f"⚠️ Warning: Model {github_ai_model} validation failed. Trying fallback models.")
            # Try a series of fallbacks in order of preference
            fallback_models = [
                GITHUB_AI_MODELS.get("GPT_4_TURBO", "openai/gpt-4-turbo"),
                GITHUB_AI_MODELS.get("LLAMA_3_1_70B", "meta/Meta-Llama-3.1-70B-Instruct"),
                GITHUB_AI_MODELS.get("CLAUDE_3_SONNET", "anthropic/claude-3-sonnet"),
                "deepseek/DeepSeek-V3-0324"  # Another reliable fallback
            ]
            
            for fallback in fallback_models:
                if self._validate_model(fallback):
                    self.model_name = fallback
                    break
            else:
                # If all fallbacks fail, use the default anyway with a warning
                self.model_name = GITHUB_AI_MODELS.get("GPT_4_TURBO", "openai/gpt-4-turbo")
                print(f"⚠️ Warning: All fallback models failed validation. Using {self.model_name} as last resort.")
            
        print(f"✅ Using GitHub AI model: {self.model_name} with Azure AI Inference SDK")
        
        # Load FAISS for header and row search
        self.header_faiss = FAISSSearcher("headers")
        self.row_faiss = FAISSSearcher("rows")

    def _generate_content(self, prompt: str) -> str:
        """Generate content using Azure AI Inference client with GitHub AI"""
        try:
            # Use Azure AI Inference client with GitHub AI configuration
            # Prepare messages
            messages = [
                SystemMessage(content="You are a data validation and analysis assistant."),
                UserMessage(content=prompt)
            ]
            
            # Configure parameters based on model type
            temperature = 0.1  # Low temperature for more deterministic outputs
            max_tokens = 2048  # Adjust as needed
            top_p = 1.0
            
            # Different models may require different parameter configurations
            if "meta" in self.model_name.lower() or "llama" in self.model_name.lower():
                # Meta Llama models typically work well with these parameters
                temperature = 0.2
                max_tokens = 4000
            
            # Set up a retry mechanism
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = self.client.complete(
                        messages=messages,
                        model=self.model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p
                    )
                    
                    # Check if we got a valid response
                    if response and hasattr(response, 'choices') and len(response.choices) > 0:
                        return response.choices[0].message.content
                    else:
                        print(f"Warning: Empty response received (attempt {retry_count + 1}/{max_retries})")
                        retry_count += 1
                except Exception as api_error:
                    print(f"API error (attempt {retry_count + 1}/{max_retries}): {api_error}")
                    retry_count += 1
                    # Wait before retrying (exponential backoff)
                    import time
                    time.sleep(2 ** retry_count)
            
            # If we reached here, all retries failed
            return "Error: Failed to generate content after multiple attempts. Please check your configuration."
            
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            # Simple fallback - return a message about the error
            return f"Error generating content. Please check your configuration and try again. Error: {str(e)}"

    def _safe_json_response(self, content: str):
        """Safely extract and parse JSON from OpenAI's response"""
        try:
            # First, check if the content contains a JSON code block
            if "```json" in content:
                json_block_start = content.find("```json") + 7
                json_block_end = content.find("```", json_block_start)
                if json_block_end > json_block_start:
                    json_content = content[json_block_start:json_block_end].strip()
                    return json.loads(json_content)
            
            # Try to parse the entire content as JSON
            return json.loads(content)
        except Exception:
            try:
                # Look for JSON array in the text
                array_start = content.find("[")
                array_end = content.rfind("]")
                if array_start >= 0 and array_end > array_start:
                    # Try to clean up the JSON by fixing common errors
                    json_str = content[array_start:array_end+1]
                    # Sometimes the JSON can have invalid line breaks or extra commas
                    return json.loads(json_str)
                
                # Look for JSON object in the text
                json_start = content.find("{")
                json_end = content.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    return json.loads(content[json_start:json_end+1])
                
                # If we can't find valid JSON, create structured errors from the text
                if content:
                    # Extract error information from the text
                    errors = []
                    # Try to identify error patterns in the text
                    if "duplicate" in content.lower():
                        errors.append({
                            "type": "duplicate_id",
                            "message": "Duplicate IDs detected in data",
                            "details": {"raw_message": content[:200]}
                        })
                    if "priority" in content.lower() and "range" in content.lower():
                        errors.append({
                            "type": "out_of_range",
                            "message": "PriorityLevel values out of valid range (1-5)",
                            "details": {"raw_message": content[:200]}
                        })
                    if "json" in content.lower() and ("invalid" in content.lower() or "malformed" in content.lower()):
                        errors.append({
                            "type": "invalid_json",
                            "message": "Invalid JSON format in AttributesJSON fields",
                            "details": {"raw_message": content[:200]}
                        })
                    if "task" in content.lower() and "invalid" in content.lower():
                        errors.append({
                            "type": "invalid_task_ids",
                            "message": "Invalid TaskIDs detected in RequestedTaskIDs",
                            "details": {"raw_message": content[:200]}
                        })
                    
                    if errors:
                        return errors
                
                return None
            except Exception as e:
                print("[ERROR] Failed to parse JSON:", e)
                return None

    def _post_process_validation_errors(self, errors):
        """
        Post-process validation errors to filter out false positives.
        
        This function examines the validation errors from the LLM and filters out:
        1. PriorityLevel values that are flagged as out of range but are actually within range (1-5)
        2. AttributesJSON fields that are flagged as invalid but are actually valid JSON
        """
        if not errors or not isinstance(errors, list):
            return errors
            
        filtered_errors = []
        
        for error in errors:
            # Skip the error if it's not a dictionary
            if not isinstance(error, dict):
                filtered_errors.append(error)
                continue
                
            # Check for false positive PriorityLevel out of range errors
            if error.get("type") == "out_of_range" and "PriorityLevel" in error.get("message", ""):
                details = error.get("details", {})
                priority_level = details.get("PriorityLevel")
                
                # If PriorityLevel is between 1-5 inclusive, it's a false positive
                if isinstance(priority_level, (int, float)) and 1 <= priority_level <= 5:
                    continue  # Skip this error, it's a false positive
            
            # Check for false positive malformed JSON errors
            elif error.get("type") == "invalid_json" and "AttributesJSON" in error.get("message", ""):
                details = error.get("details", {})
                json_value = details.get("AttributesJSON")
                
                # If we can parse it as valid JSON, it's a false positive
                if json_value and isinstance(json_value, str):
                    try:
                        json.loads(json_value)
                        continue  # Skip this error, it's a false positive
                    except json.JSONDecodeError:
                        # Only keep the error if it's actually invalid JSON
                        pass
            
            # If we made it here, keep the error
            filtered_errors.append(error)
            
        return filtered_errors

    def map_header(self, bad_header: str) -> str:
        """
        Maps a wrong/misspelled CSV header to the correct one.
        Tries GPT first; falls back to FAISS if needed.
        """
        prompt = f"""
        You are a CSV header correction assistant.
        Given this incorrect header: '{bad_header}'
        Choose the best match from the list below and ONLY return that value.

        Valid headers:
        ["TaskID", "TaskName", "Duration", "RequiredSkills", "PreferredPhases", "MaxConcurrent", "WorkerID",
        "WorkerName", "Skills", "AvailableSlots", "MaxLoadPerPhase", "WorkerGroup", "QualificationLevel",
        "ClientID", "ClientName", "PriorityLevel", "RequestedTaskIDs", "GroupTag", "AttributesJSON"]
        """
        
        try:
            response = self._generate_content(prompt).strip().replace('"', '')
            if response in self.header_faiss.entries:
                return response
        except Exception:
            pass

        # Fallback to FAISS
        return self.header_faiss.search(bad_header, top_k=1)[0]

    def validate_data(self, row_data: Union[str, Dict, List]) -> dict:
        """
        Validates a data row or set of rows for common errors and constraints.
        Accepts string JSON, dictionary, or list of dictionaries.
        Returns a list of validation errors with explanations.
        """
        # Parse the row data if it's a string
        if isinstance(row_data, str):
            try:
                parsed_data = json.loads(row_data)
            except json.JSONDecodeError:
                return {"errors": [{"type": "parse_error", "message": "Invalid JSON format"}]}
        else:
            parsed_data = row_data
            
        # Convert single row to list if needed
        if isinstance(parsed_data, dict):
            data_list = [parsed_data]
        elif isinstance(parsed_data, list):
            data_list = parsed_data
        else:
            return {"errors": [{"type": "data_error", "message": "Data must be a JSON object or array"}]}
        
        # Collect similar rows for context
        similar_rows = []
        try:
            similar_rows = self.row_faiss.search(json.dumps(data_list[0]), top_k=3)
        except Exception:
            pass
        
        prompt = f"""
        You are a data validation assistant. Analyze these rows and list ALL validation errors:
        
        Rows to validate:
        {json.dumps(data_list[:10], indent=2)}  # Send more rows for better validation
        
        Total rows in dataset: {len(data_list)}
        
        Similar valid rows for reference:
        {json.dumps(similar_rows, indent=2)}
        
        Check for these specific issues and return a JSON array of errors:
        1. Duplicate IDs (ClientID/WorkerID/TaskID) - be thorough in finding duplicates
        2. Malformed or missing data in required fields
        3. Out-of-range values (PriorityLevel should be between 1 and 5, inclusive - both 1 and 5 are valid values)
        4. Invalid JSON in AttributesJSON field (many entries have plain text instead of JSON)
        5. Invalid TaskIDs in RequestedTaskIDs - Check for format errors:
           - TaskIDs should follow the 'T' + number format
           - Examples of invalid format: "TX" (not a number)
        6. Missing required columns based on data type:
           - For clients: ClientID, ClientName, PriorityLevel
           - For tasks: TaskID, Duration, Phase
           - For workers: WorkerID, Skills
        7. Data inconsistencies (conflicting or incompatible values)
        
        Be thorough and try to catch all issues. This data has known issues with:
        - Duplicate ClientIDs
        - Invalid TaskIDs format (like TX)
        - Non-JSON text in AttributesJSON
        - PriorityLevel values out of range
        
        Format each error as:
        {{"type": "error_type", "message": "detailed message", "details": {...}}}

        ONLY return the JSON array of errors without any additional text or explanations.
        """
        
        try:
            # Get GPT's analysis
            response = self._generate_content(prompt)
            gpt_errors = self._safe_json_response(response) or []
            
            # If we couldn't parse any errors but there is response text, create a single error
            if not gpt_errors and response.strip():
                gpt_errors = [{
                    "type": "validation_summary",
                    "message": "Validation results (unparsed): " + response.strip()[:200] + "...",
                    "details": {"full_response": response.strip()}
                }]
            
            # Get FAISS validation errors - only if we have a valid object structure
            try:
                faiss_errors = self.row_faiss.validate_data(data_list)
            except Exception:
                faiss_errors = []
            
            # Combine and deduplicate errors
            all_errors = []
            seen_messages = set()
            
            for error in gpt_errors + [vars(e) for e in faiss_errors]:
                msg = error.get("message", "")
                if msg not in seen_messages:
                    all_errors.append(error)
                    seen_messages.add(msg)
            
            # Post-process and filter out false positive errors
            all_errors = self._post_process_validation_errors(all_errors)
            
            return {"errors": all_errors}
            
        except Exception as e:
            return {"errors": [{"type": "validation_error", "message": str(e)}]}
            
    def query_to_filter(self, nl_query: str) -> dict:
        """
        Convert a natural language search into a structured JSON filter.
        Handles complex queries with multiple conditions and nested logic.
        """
        prompt = f"""
        Convert this natural language query into a precise JSON filter:
        Query: "{nl_query}"

        Use these operators:
        - Comparison: "=", "!=", ">", "<", ">=", "<="
        - Lists: "includes", "excludes", "containsAll", "containsAny"
        - Text: "contains", "startsWith", "endsWith", "matches" (regex)
        - Logic: "and", "or", "not"
        
        Example queries and their JSON:
        1. "Find tasks longer than 2 hours in phase 3"
        {{
          "entity": "Tasks",
          "filter": {{
            "operator": "and",
            "conditions": [
              {{ "field": "Duration", "operator": ">", "value": 2 }},
              {{ "field": "Phase", "operator": "=", "value": 3 }}
            ]
          }}
        }}

        2. "Show workers with Python and Java skills who are available in morning slots"
        {{
          "entity": "Workers",
          "filter": {{
            "operator": "and",
            "conditions": [
              {{ 
                "field": "Skills", 
                "operator": "containsAll", 
                "value": ["Python", "Java"]
              }},
              {{
                "field": "AvailableSlots",
                "operator": "includes",
                "value": "morning"
              }}
            ]
          }}
        }}

        Convert the following query using similar structure:
        "{nl_query}"

        ONLY return the JSON, without any additional explanation or formatting.
        """
        response = self._safe_json_response(self._generate_content(prompt))
        
        # Validate the filter structure
        if response and "filter" in response:
            try:
                # Use FAISS to find similar valid queries for reference
                similar_queries = self.row_faiss.search(nl_query, top_k=2)
                
                # Ask GPT to verify and refine the filter
                verification_prompt = f"""
                Verify and improve this filter if needed:
                {json.dumps(response, indent=2)}

                Similar successful queries for reference:
                {json.dumps(similar_queries, indent=2)}

                Return the verified/improved filter in the same JSON format.
                ONLY return the JSON, without any additional explanation or formatting.
                """
                improved_response = self._safe_json_response(
                    self._generate_content(verification_prompt)
                )
                if improved_response:
                    return improved_response
            except Exception:
                pass
                
        return response

    def modify_data(self, nl_command: str) -> dict:
        """
        Convert a natural language update command into a structured modification.
        Handles complex modifications including multi-field updates and conditions.
        """
        prompt = f"""
        Convert this natural language instruction into a precise modification command:
        Instruction: "{nl_command}"

        Support these modification types:
        1. Single field update
        2. Multi-field update
        3. Conditional update
        4. Batch update
        5. Complex value transformations
        
        Examples:
        1. "Change task T1's duration to 4 hours"
        {{
          "type": "update",
          "entity": "Task",
          "target": {{ "id": "T1" }},
          "changes": [
            {{ "field": "Duration", "value": 4 }}
          ]
        }}

        2. "For all tasks in phase 3, increase priority by 1 and add Python to required skills"
        {{
          "type": "batchUpdate",
          "entity": "Task",
          "condition": {{ "field": "Phase", "operator": "=", "value": 3 }},
          "changes": [
            {{ 
              "field": "PriorityLevel", 
              "operation": "increment",
              "value": 1
            }},
            {{
              "field": "RequiredSkills",
              "operation": "append",
              "value": "Python"
            }}
          ]
        }}

        Convert the following instruction using similar structure:
        "{nl_command}"

        ONLY return the JSON, without any additional explanation or formatting.
        """
        
        response = self._safe_json_response(self._generate_content(prompt))
        
        # Validate the modification command
        if response:
            try:
                # Get similar examples for validation
                similar_examples = self.row_faiss.search(nl_command, top_k=2)
                
                # Ask GPT to validate and ensure data consistency
                validation_prompt = f"""
                Validate this modification command for data consistency:
                {json.dumps(response, indent=2)}

                Consider:
                1. Data type compatibility
                2. Value range constraints
                3. Reference integrity
                4. Business rule compliance

                Similar examples for reference:
                {json.dumps(similar_examples, indent=2)}

                Return the validated/corrected command in the same JSON format.
                ONLY return the JSON, without any additional explanation or formatting.
                """
                
                validated_response = self._safe_json_response(
                    self._generate_content(validation_prompt)
                )
                if validated_response:
                    return validated_response
                    
            except Exception:
                pass
                
        return response

    def suggest_correction(self, error_row: str) -> dict:
        """
        Use FAISS and GPT to suggest intelligent corrections for invalid data.
        Handles multiple error types and provides explanation for corrections.
        """
        # Find multiple similar valid rows for better context
        similar_rows = self.row_faiss.search(error_row, top_k=3)
        
        # Get validation errors
        validation_result = self.validate_data(error_row)
        
        prompt = f"""
        Analyze and correct this problematic row:
        {error_row}

        Validation Errors Found:
        {json.dumps(validation_result.get('errors', []), indent=2)}

        Similar Valid Examples:
        {json.dumps(similar_rows, indent=2)}

        Provide corrections following these rules:
        1. Fix all validation errors
        2. Maintain data consistency
        3. Preserve valid fields
        4. Follow business rules
        5. Explain each correction

        Return in this format:
        {{
          "corrected_data": {{ corrected row as JSON }},
          "corrections": [
            {{
              "field": "field_name",
              "old_value": "previous value",
              "new_value": "corrected value",
              "reason": "explanation for change"
            }}
          ],
          "confidence_score": 0.95  // How confident in the corrections
        }}

        ONLY return the JSON, without any additional explanation or formatting.
        """
        
        response = self._safe_json_response(self._generate_content(prompt))
        
        # Validate the suggested corrections
        if response and "corrected_data" in response:
            # Verify the corrections fix the issues
            validation_after = self.validate_data(json.dumps(response["corrected_data"]))
            if not validation_after.get("errors"):
                return response
            
            # If issues remain, try one more time with more context
            retry_prompt = f"""
            The suggested corrections still have these issues:
            {json.dumps(validation_after["errors"], indent=2)}

            Please provide a new correction that resolves ALL validation issues.
            Use the same JSON response format as before.
            ONLY return the JSON, without any additional explanation or formatting.
            """
            final_response = self._safe_json_response(
                self._generate_content(retry_prompt)
            )
            if final_response:
                return final_response
                
        return response

    def nl_to_rule(self, rule_text: str) -> dict:
        """
        Convert natural language rules into structured JSON rules.
        Handles complex business rules, constraints, and dependencies.
        """
        prompt = f"""
        Convert this natural language rule into a precise JSON rule definition:
        "{rule_text}"

        Support these rule types:
        1. Task Dependencies
           - Co-run requirements
           - Sequential ordering
           - Time window constraints
           
        2. Resource Constraints
           - Worker load limits
           - Skill requirements
           - Phase capacity
           
        3. Business Rules
           - Priority handling
           - Client preferences
           - Quality requirements

        Examples:
        1. "Tasks T1 and T2 must run together in the morning phase"
        {{
          "type": "taskDependency",
          "subtype": "coRun",
          "rules": [
            {{
              "tasks": ["T1", "T2"],
              "constraints": {{
                "phase": "morning",
                "enforceSync": true
              }}
            }}
          ]
        }}

        2. "Workers in Group A can handle max 3 tasks per phase and need Python certification"
        {{
          "type": "resourceConstraint",
          "subtype": "workerCapacity",
          "rules": [
            {{
              "target": {{
                "group": "Group A"
              }},
              "constraints": [
                {{
                  "type": "loadLimit",
                  "maxTasksPerPhase": 3
                }},
                {{
                  "type": "skillRequirement",
                  "skills": ["Python"],
                  "certificationRequired": true
                }}
              ]
            }}
          ]
        }}

        Convert the following rule using similar structure:
        "{rule_text}"

        ONLY return the JSON, without any additional explanation or formatting.
        """
        
        response = self._safe_json_response(self._generate_content(prompt))
        
        # Validate and enhance the rule
        if response:
            try:
                # Find similar existing rules
                similar_rules = self.row_faiss.search(rule_text, top_k=2)
                
                # Ask GPT to validate rule consistency
                validation_prompt = f"""
                Validate this rule for consistency and completeness:
                {json.dumps(response, indent=2)}

                Consider:
                1. Rule conflicts
                2. Circular dependencies
                3. Resource feasibility
                4. Business logic consistency

                Similar rules for reference:
                {json.dumps(similar_rules, indent=2)}

                Return the validated/enhanced rule in the same JSON format.
                ONLY return the JSON, without any additional explanation or formatting.
                """
                
                validated_response = self._safe_json_response(
                    self._generate_content(validation_prompt)
                )
                if validated_response:
                    return validated_response
                    
            except Exception:
                pass
                
        return response

    def rule_recommendations(self, context: str) -> dict:
        """
        Analyze dataset patterns to suggest intelligent business rules.
        Uses historical data and current context to recommend optimal rules.
        """
        # First, let's analyze the context using FAISS for pattern recognition
        similar_contexts = self.row_faiss.search(context, top_k=5)
        
        prompt = f"""
        Analyze this operational context and suggest optimal rules:
        {context}

        Similar historical contexts and their patterns:
        {json.dumps(similar_contexts, indent=2)}

        Provide comprehensive rule recommendations considering:

        1. Workload Optimization
           - Task grouping patterns
           - Resource utilization
           - Phase balancing
           
        2. Quality Assurance
           - Skill matching
           - Certification requirements
           - Quality control points
           
        3. Efficiency Rules
           - Common bottlenecks
           - Parallel processing
           - Resource allocation
           
        4. Risk Management
           - Dependency management
           - Fallback procedures
           - Error prevention
           
        Return in this format:
        {{
          "recommendations": [
            {{
              "category": "category_name",
              "rules": [
                {{
                  "type": "rule_type",
                  "priority": 1-5,
                  "configuration": {{ rule details }},
                  "impact": {{
                    "efficiency": 0.8,
                    "quality": 0.9,
                    "risk": 0.7
                  }},
                  "rationale": "explanation"
                }}
              ]
            }}
          ],
          "meta": {{
            "confidence_score": 0.95,
            "data_coverage": 0.85,
            "implementation_complexity": "medium"
          }}
        }}

        ONLY return the JSON, without any additional explanation or formatting.
        """
        
        response = self._safe_json_response(self._generate_content(prompt))
        
        # Validate and prioritize recommendations
        if response and "recommendations" in response:
            try:
                # Get validation results for the context
                validation_results = self.validate_data(context)
                
                # Ask GPT to refine recommendations based on validation
                refinement_prompt = f"""
                Refine these rule recommendations considering these validation issues:
                {json.dumps(validation_results.get('errors', []), indent=2)}

                Current recommendations:
                {json.dumps(response, indent=2)}

                Adjust recommendations to:
                1. Address validation issues
                2. Prevent similar problems
                3. Optimize for current context
                4. Ensure implementation feasibility

                Use the same JSON format for the response.
                ONLY return the JSON, without any additional explanation or formatting.
                """
                
                refined_response = self._safe_json_response(
                    self._generate_content(refinement_prompt)
                )
                if refined_response:
                    return refined_response
                    
            except Exception:
                pass
                
        return response

    def generate_priority_profile(self, summary: str) -> dict:
        """
        Generate intelligent priority profiles based on operational context.
        Uses historical data and current state to optimize weight distribution.
        """
        # Get historical context from FAISS
        similar_summaries = self.row_faiss.search(summary, top_k=3)
        
        prompt = f"""
        Analyze this operational summary and generate an optimal priority profile:
        {summary}

        Historical context for reference:
        {json.dumps(similar_summaries, indent=2)}

        Consider these factors:
        1. Resource Utilization
           - Worker capacity
           - Skill distribution
           - Phase load balance
           
        2. Business Priorities
           - Client SLAs
           - Task urgency
           - Quality requirements
           
        3. Operational Efficiency
           - Task dependencies
           - Resource availability
           - Processing capacity
           
        4. Risk Management
           - Deadline compliance
           - Quality assurance
           - Error prevention

        Return in this format:
        {{
          "priority_weights": {{
            "client_priority": 0.3,
            "task_urgency": 0.2,
            "resource_efficiency": 0.15,
            "quality_assurance": 0.15,
            "load_balance": 0.1,
            "risk_management": 0.1
          }},
          "factor_thresholds": {{
            "max_concurrent_tasks": 5,
            "min_skill_coverage": 0.8,
            "quality_score_threshold": 0.9
          }},
          "scaling_factors": {{
            "deadline_pressure": 1.2,
            "resource_availability": 0.9
          }},
          "meta": {{
            "confidence_score": 0.95,
            "adaptation_rate": 0.1,
            "review_interval": "4h"
          }}
        }}

        Ensure all weights sum to 1.0 and provide realistic thresholds.
        ONLY return the JSON, without any additional explanation or formatting.
        """
        
        response = self._safe_json_response(self._generate_content(prompt))
        
        # Validate and optimize the profile
        if response and "priority_weights" in response:
            try:
                # Get current system state
                validation_results = self.validate_data(summary)
                
                # Ask GPT to optimize the profile
                optimization_prompt = f"""
                Optimize this priority profile considering current system state:
                {json.dumps(response, indent=2)}

                System validation results:
                {json.dumps(validation_results.get('errors', []), indent=2)}

                Adjust the profile to:
                1. Address current issues
                2. Optimize resource usage
                3. Maintain SLA compliance
                4. Balance competing priorities

                Use the same JSON format for the response.
                ONLY return the JSON, without any additional explanation or formatting.
                """
                
                optimized_response = self._safe_json_response(
                    self._generate_content(optimization_prompt)
                )
                if optimized_response:
                    return optimized_response
                    
            except Exception:
                pass
                
        return response

    def analyze_file(self, file_path: str) -> dict:
        """
        Analyzes a CSV or Excel file to identify all errors and provide detailed fix recommendations.
        Supports multiple error types and common data issues.
        
        Args:
            file_path: Path to the CSV or Excel file to analyze
            
        Returns:
            A dictionary with comprehensive error analysis and fix recommendations
        """
        print(f"Starting analysis of file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}"
            }
        
        # Determine file type
        file_ext = Path(file_path).suffix.lower()
        
        try:
            # Load the file based on extension
            if file_ext == '.csv':
                # Try different encodings and delimiters for CSV
                try:
                    print("Attempting to read CSV with utf-8 encoding...")
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        print("Attempting to read CSV with latin1 encoding...")
                        df = pd.read_csv(file_path, encoding='latin1')
                    except:
                        print("Attempting to read CSV with cp1252 encoding...")
                        df = pd.read_csv(file_path, encoding='cp1252')
                except pd.errors.ParserError:
                    # Try with different delimiters
                    print("Attempting to read CSV with semicolon delimiter...")
                    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
            elif file_ext in ['.xlsx', '.xls']:
                print("Reading Excel file...")
                df = pd.read_excel(file_path)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported file type: {file_ext}. Please provide a CSV or Excel file."
                }
                
            print(f"File loaded successfully: {len(df)} rows, {len(df.columns)} columns")
                
            # Basic file statistics
            file_stats = {
                "rows": len(df),
                "columns": len(df.columns),
                "file_type": "CSV" if file_ext == '.csv' else "Excel",
                "file_name": os.path.basename(file_path),
                "file_size": f"{os.path.getsize(file_path) / 1024:.2f} KB",
                "column_names": list(df.columns)
            }
            
            # Perform initial data analysis
            print("Analyzing data quality issues...")
            
            missing_values = df.isnull().sum().to_dict()
            duplicate_rows = df.duplicated().sum()
            
            # Check for basic column issues
            column_issues = {}
            for col in df.columns:
                # Check for spaces or special characters in column names
                if re.search(r'[\s\W]', col):
                    column_issues[col] = "Contains spaces or special characters"
                # Check for duplicate column names
                if list(df.columns).count(col) > 1:
                    column_issues[col] = "Duplicate column name"
            
            print(f"Found {sum(1 for v in missing_values.values() if v > 0)} columns with missing values")
            print(f"Found {duplicate_rows} duplicate rows")
            print(f"Found {len(column_issues)} column name issues")
            
            # Sample data for LLM analysis
            sample_rows = df.head(5).to_dict(orient='records')
            
            # Perform advanced checks that we know about
            advanced_issues = []
            
            # Check for duplicate IDs
            for col in df.columns:
                if col.endswith('ID'):
                    duplicates = df[col].duplicated().sum()
                    if duplicates > 0:
                        print(f"Found {duplicates} duplicate values in {col}")
                        advanced_issues.append({
                            "type": "duplicate_id",
                            "location": col,
                            "description": f"{duplicates} duplicate values detected in {col}",
                            "impact": "Duplicate IDs can cause reference integrity issues",
                            "fix": f"Fix duplicate {col} values to ensure uniqueness",
                            "severity": "critical"
                        })
                
            # Check for values outside expected ranges
            if 'PriorityLevel' in df.columns and df['PriorityLevel'].dtype in ['int64', 'float64']:
                invalid_priority = df[~df['PriorityLevel'].between(1, 5)].shape[0]
                if invalid_priority > 0:
                    print(f"Found {invalid_priority} rows with PriorityLevel outside valid range (1-5)")
                    advanced_issues.append({
                        "type": "out_of_range",
                        "location": "PriorityLevel",
                        "description": f"{invalid_priority} rows have PriorityLevel outside valid range (1-5)",
                        "impact": "Invalid priority levels can affect scheduling and processing",
                        "fix": "Correct PriorityLevel values to be between 1 and 5",
                        "severity": "critical"
                    })
                
            # Check for invalid JSON in columns with 'JSON' in the name
            for col in df.columns:
                if 'JSON' in col:
                    def is_valid_json(text):
                        try:
                            if isinstance(text, str):
                                json.loads(text)
                                return True
                        except:
                            return False
                        return isinstance(text, str) and bool(text.strip())
                    
                    invalid_json_count = sum(~df[col].apply(is_valid_json))
                    if invalid_json_count > 0:
                        print(f"Found {invalid_json_count} rows with invalid JSON in {col}")
                        advanced_issues.append({
                            "type": "invalid_json",
                            "location": col,
                            "description": f"{invalid_json_count} rows have invalid JSON in {col}",
                            "impact": "Invalid JSON can cause parsing errors and data loss",
                            "fix": f"Convert plain text in {col} to valid JSON format",
                            "severity": "critical"
                        })
                
            # Check for invalid TaskIDs in RequestedTaskIDs
            if 'RequestedTaskIDs' in df.columns:
                def has_invalid_task_id(task_list):
                    if not isinstance(task_list, str):
                        return False
                    tasks = task_list.split(',')
                    for task in tasks:
                        task = task.strip()
                        if not task:
                            continue
                        if not (task.startswith('T') and task[1:].isdigit()):
                            return True
                        elif task.startswith('T') and task[1:].isdigit() and int(task[1:]) > 50:
                            return True
                    return False
                
                invalid_tasks_count = sum(df['RequestedTaskIDs'].apply(has_invalid_task_id))
                if invalid_tasks_count > 0:
                    print(f"Found {invalid_tasks_count} rows with invalid TaskIDs in RequestedTaskIDs")
                    advanced_issues.append({
                        "type": "invalid_task_ids",
                        "location": "RequestedTaskIDs",
                        "description": f"{invalid_tasks_count} rows have invalid TaskIDs in RequestedTaskIDs",
                        "impact": "Invalid TaskIDs can cause reference errors and processing failures",
                        "fix": "Remove or correct invalid TaskIDs (format should be T1-T50)",
                        "severity": "critical"
                    })
                    
            # Check for text case inconsistency
            if 'GroupTag' in df.columns:
                # Get unique values
                unique_groups = df['GroupTag'].unique()
                # Check if there's a mix of uppercase and lowercase versions of the same value
                lowercase_groups = [g.lower() if isinstance(g, str) else g for g in unique_groups]
                if len(set(lowercase_groups)) < len(set(unique_groups)):
                    print(f"Found inconsistent text case in GroupTag values")
                    advanced_issues.append({
                        "type": "inconsistent_text_case",
                        "location": "GroupTag",
                        "description": "GroupTag values have inconsistent text case (e.g., 'GroupA' and 'groupa')",
                        "impact": "Inconsistent text case can cause grouping and filtering issues",
                        "fix": "Standardize text case to uppercase or lowercase",
                        "severity": "warning"
                    })
                    
            # Check for whitespace issues
            if 'ClientName' in df.columns:
                # Count rows where trimming would change the value
                whitespace_issues = sum(df['ClientName'].apply(
                    lambda x: isinstance(x, str) and (x.strip() != x or x != x.strip())
                ))
                if whitespace_issues > 0:
                    print(f"Found {whitespace_issues} rows with leading/trailing whitespace in ClientName")
                    advanced_issues.append({
                        "type": "leading_trailing_whitespace",
                        "location": "ClientName",
                        "description": f"{whitespace_issues} rows have leading or trailing whitespace in ClientName",
                        "impact": "Extra whitespace can cause matching and display issues",
                        "fix": "Trim whitespace from ClientName values",
                        "severity": "warning"
                    })
                    
            # Handle more specific data type issues
            for col in df.columns:
                # Check date columns
                if 'date' in col.lower():
                    # Attempt to convert to datetime if not already
                    if df[col].dtype != 'datetime64[ns]':
                        try:
                            pd.to_datetime(df[col], errors='raise')
                            print(f"Found date column {col} with incorrect data type")
                            advanced_issues.append({
                                "type": "incorrect_data_type",
                                "location": col,
                                "description": f"{col} contains date values but is not a datetime type",
                                "impact": "Date operations may not work correctly",
                                "fix": f"Convert {col} to datetime type"
                            })
                        except:
                            pass
            
            # Prepare analysis prompt for LLM
            print("Generating LLM analysis prompt...")
            
            prompt = f"""
            You are a data analysis expert. Analyze this {file_ext[1:].upper()} file and identify ALL possible errors and data quality issues:
            
            File Statistics:
            {json.dumps(file_stats, indent=2)}
            
            Column Names:
            {list(df.columns)}
            
            Column Issues Already Identified:
            {json.dumps(column_issues, indent=2)}
            
            Missing Values by Column:
            {json.dumps({k: int(v) for k, v in missing_values.items() if v > 0}, indent=2)}
            
            Duplicate Rows: {duplicate_rows}
            
            Advanced Issues Already Detected:
            {json.dumps([{
                "type": issue["type"],
                "location": issue["location"],
                "description": issue["description"]
            } for issue in advanced_issues], indent=2)}
            
            Sample Data (first 5 rows):
            {json.dumps(sample_rows, indent=2)}
            
            Perform a comprehensive analysis of ALL possible data quality issues, including but not limited to:
            
            1. Data Type Issues:
               - Numeric columns with text values
               - Date columns with invalid formats
               - Categorical columns with inconsistent values
               
            2. Value Range Issues:
               - Negative values in positive-only fields
               - Values outside of acceptable ranges
               - Outliers that may indicate data entry errors
               
            3. Formatting Issues:
               - Inconsistent date formats
               - Inconsistent text case (mixed upper/lower)
               - Leading/trailing whitespace
               - Special characters or control characters
               
            4. Referential Integrity Issues:
               - Invalid IDs or references
               - Incomplete hierarchical data
               
            5. Business Rule Violations:
               - Conditional value constraints
               - Cross-field validation issues
               - Logical contradictions
               
            6. Structural Issues:
               - Merged cells (Excel)
               - Hidden rows or columns
               - Inconsistent number of columns across rows
               
            For EACH identified issue:
            1. Describe the problem in detail
            2. Specify where it occurs (column/row)
            3. Explain the potential impact
            4. Provide a SPECIFIC fix recommendation with exact steps or code
            
            Return your analysis as a structured JSON object with this format:
            {{
                "issues": [
                    {{
                        "type": "issue_type", 
                        "location": "column_name or row_range",
                        "description": "detailed description of the problem",
                        "impact": "explanation of the business impact",
                        "fix": "recommendation to fix the issue",
                        "fix_code": "Python code to fix the issue (if possible)"
                    }}
                ]
            }}
            
            ONLY return the JSON, without any additional text or explanations.
            """
            
            # Get LLM response
            print("Sending request to LLM for analysis...")
            llm_response = self._generate_content(prompt)
            llm_analysis = self._safe_json_response(llm_response)
            
            # Combine our detected issues with LLM issues
            if not llm_analysis or "issues" not in llm_analysis:
                print("LLM analysis failed or returned no issues. Using only our detected issues.")
                llm_analysis = {
                    "issues": []
                }
            else:
                print(f"LLM found {len(llm_analysis['issues'])} additional issues")
                
                # Validate LLM issues to filter out invalid or inconsistent ones
                validated_issues = []
                for issue in llm_analysis.get("issues", []):
                    # Validate issue structure
                    if not isinstance(issue, dict) or "type" not in issue or "location" not in issue:
                        continue
                        
                    # Filter out issues that don't match our expected issue types
                    issue_type = issue.get("type", "").lower()
                    valid_types = [
                        "duplicate_id", "out_of_range", "invalid_json", "invalid_task_ids", 
                        "missing_values", "column_name", "duplicate_rows", "formatting",
                        "data_type", "inconsistent_text_case", "leading_trailing_whitespace"
                    ]
                    
                    # Only include issues with recognized types or with strong evidence
                    if issue_type in valid_types or "description" in issue and issue["description"]:
                        # Add severity if not present
                        if "severity" not in issue:
                            # Determine severity based on type and description
                            impact = issue.get("impact", "")
                            # Try to infer count from description
                            count = None
                            desc = issue.get("description", "")
                            if desc:
                                # Try to extract a number from the description
                                matches = re.findall(r'(\d+)\s*rows', desc)
                                if matches:
                                    try:
                                        count = int(matches[0])
                                    except:
                                        pass
                            
                            issue["severity"] = self._determine_issue_severity(issue_type, count, impact)
                            
                        validated_issues.append(issue)
                
                llm_analysis["issues"] = validated_issues
                print(f"After validation: {len(validated_issues)} additional issues")
            
            # Start with our advanced issues
            all_issues = advanced_issues.copy()
            
            # Add basic issues we detected
            for col, issue in column_issues.items():
                all_issues.append({
                    "type": "column_name",
                    "location": col,
                    "description": issue,
                    "impact": "May cause problems when referencing columns programmatically",
                    "fix": f"Rename column '{col}' to remove spaces and special characters",
                    "severity": "warning"
                })
            
            for col, count in missing_values.items():
                if count > 0:
                    all_issues.append({
                        "type": "missing_values",
                        "location": col,
                        "description": f"{count} missing values detected",
                        "impact": "Incomplete data may affect analysis accuracy",
                        "fix": "Fill missing values with appropriate defaults or remove rows",
                        "severity": self._determine_issue_severity("missing_values", count=count)
                    })
            
            if duplicate_rows > 0:
                all_issues.append({
                    "type": "duplicate_rows",
                    "location": "entire file",
                    "description": f"{duplicate_rows} duplicate rows detected",
                    "impact": "Duplicates may skew analysis and results",
                    "fix": "Remove duplicate rows using df.drop_duplicates()",
                    "severity": self._determine_issue_severity("duplicate_rows", count=duplicate_rows)
                })
            
            # Add LLM detected issues, avoiding duplicates
            seen_issues = set()
            for issue in all_issues:
                key = f"{issue.get('type')}_{issue.get('location')}"
                seen_issues.add(key)
            
            for issue in llm_analysis.get("issues", []):
                key = f"{issue.get('type')}_{issue.get('location')}"
                if key not in seen_issues:
                    all_issues.append(issue)
                    seen_issues.add(key)
            
            # Update the llm_analysis with all issues
            llm_analysis["issues"] = all_issues
            
            # Now generate fix code for each issue
            print("Generating fix code for each issue...")
            
            for issue in all_issues:
                if not issue.get("fix_code"):
                    # Generate Python code to fix the issue
                    try:
                        # Special handling for known issue types
                        if issue.get("type") == "duplicate_id" and issue.get("location"):
                            col = issue.get("location")
                            issue["fix_code"] = f"""# Find duplicated {col} values
duplicate_indices = df[df['{col}'].duplicated(keep='first')].index

# Add a suffix to make them unique
for idx in duplicate_indices:
    df.loc[idx, '{col}'] = f"{{df.loc[idx, '{col}']}}_dup"
"""
                        elif issue.get("type") == "out_of_range" and issue.get("location") == "PriorityLevel":
                            issue["fix_code"] = """# Fix out-of-range PriorityLevel values
df['PriorityLevel'] = df['PriorityLevel'].apply(lambda x: min(max(x, 1), 5) if pd.notnull(x) else 3)
"""
                        elif issue.get("type") == "invalid_json" and "JSON" in issue.get("location", ""):
                            col = issue.get("location")
                            issue["fix_code"] = f"""# Fix invalid JSON in {col}
# First, infer the schema from valid JSON in the column
def infer_json_schema(df, column):
    schema = {{}}
    valid_jsons = []
    
    # Collect all valid JSON objects
    for val in df[column]:
        if isinstance(val, str):
            try:
                json_obj = json.loads(val)
                if isinstance(json_obj, dict):
                    valid_jsons.append(json_obj)
            except:
                pass
    
    if not valid_jsons:
        return {{"notes": ""}}
    
    # Collect keys and their frequencies
    key_freq = {{}}
    key_types = {{}}
    
    for json_obj in valid_jsons:
        for key, value in json_obj.items():
            key_freq[key] = key_freq.get(key, 0) + 1
            value_type = type(value).__name__
            if key not in key_types:
                key_types[key] = {{value_type: 1}}
            else:
                key_types[key][value_type] = key_types[key].get(value_type, 0) + 1
    
    # Determine most frequent type for each key
    for key, type_counts in key_types.items():
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # Create default values based on type
        if most_common_type == 'str':
            schema[key] = ""
        elif most_common_type == 'int':
            schema[key] = 0
        elif most_common_type == 'float':
            schema[key] = 0.0
        elif most_common_type == 'bool':
            schema[key] = False
        elif most_common_type == 'dict':
            schema[key] = {{}}
        elif most_common_type == 'list':
            schema[key] = []
    
    # Sort keys by frequency
    sorted_keys = sorted(key_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Final schema with most common keys
    final_schema = {{}}
    for key, _ in sorted_keys:
        final_schema[key] = schema[key]
        
    return final_schema

# Infer schema from valid JSON
json_schema = infer_json_schema(df, '{col}')
print(f"Inferred JSON schema: {{json_schema}}")

def fix_json_with_schema(value):
    if pd.isna(value):
        return json.dumps(json_schema)
        
    if not isinstance(value, str):
        # For non-string values, use schema
        new_obj = json_schema.copy()
        string_keys = [k for k, v in json_schema.items() if isinstance(v, str)]
        if string_keys:
            new_obj[string_keys[0]] = str(value)
        elif json_schema:
            first_key = list(json_schema.keys())[0]
            new_obj[first_key] = str(value)
        else:
            new_obj = {{"notes": str(value)}}
        return json.dumps(new_obj)
    
    # Check if already valid JSON
    try:
        json.loads(value)
        return value
    except:
        # Try to extract key-value pairs if it looks like JSON
        new_obj = json_schema.copy()
        
        # Check for partial JSON or key-value format
        if (':' in value) or ('{' in value and '}' in value) or ('"' in value):
            try:
                # For values that look like JSON with missing brackets
                if value.strip().startswith('"') and not value.strip().startswith('{{'):
                    value = '{{' + value + '}}'
                
                # For values with malformed JSON
                if '{{' in value and '}}' in value:
                    # Extract content between {{ and }}
                    import re
                    json_content = re.search(r'\{{(.*?)\}}', value)
                    if json_content:
                        extracted = '{{' + json_content.group(1) + '}}'
                        try:
                            parsed = json.loads(extracted)
                            return json.dumps(parsed)
                        except:
                            pass
                
                # Try parsing key-value pairs
                parts = [p.strip() for p in value.split(',')]
                for part in parts:
                    if ':' in part:
                        k, v = part.split(':', 1)
                        k = k.strip().strip('"{{}}').strip()
                        v = v.strip().strip('"{{}}').strip()
                        
                        # Try to convert to appropriate type
                        if v.lower() == 'true':
                            new_obj[k] = True
                        elif v.lower() == 'false':
                            new_obj[k] = False
                        elif v.isdigit():
                            new_obj[k] = int(v)
                        elif v.replace('.', '', 1).isdigit():
                            new_obj[k] = float(v)
                        else:
                            new_obj[k] = v
            except:
                # If key-value extraction fails, use plain text approach
                pass
        else:
            # For plain text, try to map to common schema fields
            if 'notes' in json_schema:
                new_obj['notes'] = str(value)
            elif 'description' in json_schema:
                new_obj['description'] = str(value)
            elif 'text' in json_schema:
                new_obj['text'] = str(value)
            elif 'location' in json_schema and len(str(value).split()) <= 3:
                new_obj['location'] = str(value)
            else:
                string_keys = [k for k, v in json_schema.items() if isinstance(v, str)]
                if string_keys:
                    new_obj[string_keys[0]] = str(value)
                else:
                    first_key = list(json_schema.keys())[0] if json_schema else 'notes'
                    new_obj[first_key] = str(value)
        
        return json.dumps(new_obj)

df['{col}'] = df['{col}'].apply(fix_json_with_schema)
"""
                        elif issue.get("type") == "invalid_task_ids" and issue.get("location") == "RequestedTaskIDs":
                            issue["fix_code"] = """# Fix invalid TaskIDs in RequestedTaskIDs
def fix_task_ids(task_list):
    if not isinstance(task_list, str):
        return ''
    tasks = task_list.split(',')
    valid_tasks = []
    for task in tasks:
        task = task.strip()
        # Keep only valid tasks (T1-T50)
        if task.startswith('T') and task[1:].isdigit() and 1 <= int(task[1:]) <= 50:
            valid_tasks.append(task)
    return ','.join(valid_tasks)

df['RequestedTaskIDs'] = df['RequestedTaskIDs'].apply(fix_task_ids)
"""
                        elif issue.get("type") == "missing_values" and issue.get("location"):
                            col = issue.get("location")
                            # Generate appropriate default value based on column name
                            default_value = "0"
                            if "name" in col.lower() or "text" in col.lower() or "description" in col.lower():
                                default_value = "'Unknown'"
                            elif "date" in col.lower():
                                default_value = "pd.Timestamp('today')"
                            elif "json" in col.lower() or "JSON" in col:
                                default_value = "'{}'";
                                
                            issue["fix_code"] = f"""# Fill missing values in {col}
df['{col}'] = df['{col}'].fillna({default_value})
"""
                        elif issue.get("type") == "duplicate_rows":
                            issue["fix_code"] = """# Remove duplicate rows
df = df.drop_duplicates()
"""
                        else:
                            # Use LLM to generate fix code for other issue types
                            fix_prompt = f"""
                            Generate Python code to fix this data issue:
                            
                            Issue Type: {issue.get('type')}
                            Location: {issue.get('location')}
                            Description: {issue.get('description')}
                            Current Fix Recommendation: {issue.get('fix')}
                            
                            File type: {file_ext}
                            Sample data structure:
                            {json.dumps(sample_rows[:2], indent=2)}
                            
                            Provide ONLY the Python code to fix this specific issue. Use pandas and assume the dataframe is loaded as 'df'.
                            The code should be a complete solution that actually fixes the problem, not pseudocode.
                            DO NOT include explanations, just the code.
                            """
                            
                            fix_code = self._generate_content(fix_prompt).strip()
                            # Clean up code block markers if present
                            if fix_code.startswith("```python"):
                                fix_code = fix_code[10:]
                            if fix_code.endswith("```"):
                                fix_code = fix_code[:-3]
                            
                            issue["fix_code"] = fix_code.strip()
                    except Exception as e:
                        print(f"Error generating fix code for {issue.get('type')} in {issue.get('location')}: {str(e)}")
                        # Leave fix_code empty if there was an error
            
            # Create the final result
            print("Building final analysis result...")
            
            # Separate critical issues from warnings
            critical_issues = [issue for issue in all_issues if issue.get("severity") == "critical"]
            warnings = [issue for issue in all_issues if issue.get("severity") == "warning"]
            
            result = {
                "status": "success",
                "file_info": file_stats,
                "summary": {
                    "total_issues": len(all_issues),
                    "critical_issues": len(critical_issues),
                    "warnings": len(warnings),
                    "missing_values": sum(1 for v in missing_values.values() if v > 0),
                    "duplicate_rows": int(duplicate_rows),
                    "column_issues": len(column_issues),
                    "issue_types": {}
                },
                "detailed_analysis": {
                    "critical_issues": critical_issues,
                    "warnings": warnings
                },
                "fixes": {}
            }
            
            # Summarize issue types for quick reference
            for issue in all_issues:
                issue_type = issue.get("type", "unknown")
                if issue_type in result["summary"]["issue_types"]:
                    result["summary"]["issue_types"][issue_type] += 1
                else:
                    result["summary"]["issue_types"][issue_type] = 1
            
            # Generate fix instructions for each issue
            if all_issues:
                critical_fixes = {}
                warning_fixes = {}
                
                for i, issue in enumerate(all_issues):
                    issue_id = f"issue_{i+1}"
                    fix_details = {
                        "type": issue.get("type", "unknown"),
                        "description": issue.get("description", "No description"),
                        "location": issue.get("location", "Unknown"),
                        "impact": issue.get("impact", "Unknown impact"),
                        "fix_recommendation": issue.get("fix", "No fix recommendation"),
                        "fix_code": issue.get("fix_code", ""),
                        "severity": issue.get("severity", "warning"),
                        "before_after_examples": []
                    }
                    
                    # Generate before/after examples for common issue types
                    try:
                        if issue.get("type") == "duplicate_id" and issue.get("location"):
                            # Find example of duplicates
                            col = issue.get("location")
                            if col in df.columns:
                                dupes = df[df[col].duplicated(keep=False)].sort_values(col).head(2)
                                if not dupes.empty:
                                    examples = dupes.to_dict(orient="records")
                                    if len(examples) >= 2:
                                        before = examples[0]
                                        after = examples[1].copy()
                                        after[col] = f"{after[col]}_dup"  # Show fixed version
                                        fix_details["before_after_examples"].append({
                                            "before": before,
                                            "after": after,
                                            "explanation": f"Duplicate {col} value '{before[col]}' was made unique by adding '_dup' suffix"
                                        })
                        
                        elif issue.get("type") == "invalid_json" and issue.get("location"):
                            # Find example of invalid JSON
                            col = issue.get("location")
                            if col in df.columns:
                                # Infer schema from valid JSON rows
                                json_schema = self._infer_json_schema(df, col)
                                
                                # Find the first row with invalid JSON
                                for idx, val in df[col].items():
                                    try:
                                        if isinstance(val, str):
                                            json.loads(val)
                                    except:
                                        row = df.loc[idx].to_dict()
                                        fixed_row = row.copy()
                                        
                                        # Apply schema-based fixing
                                        new_obj = json_schema.copy()
                                        
                                        # Handle partial JSON or key-value format
                                        if isinstance(val, str) and (':' in val or '{' in val or '"' in val):
                                            try:
                                                # For values that look like JSON with missing brackets
                                                if val.strip().startswith('"') and not val.strip().startswith('{"'):
                                                    val = '{' + val + '}'
                                                
                                                # Extract content between { and }
                                                if '{' in val and '}' in val:
                                                    json_content = re.search(r'\{(.*?)\}', val)
                                                    if json_content:
                                                        extracted = '{' + json_content.group(1) + '}'
                                                        try:
                                                            parsed = json.loads(extracted)
                                                            fixed_row[col] = json.dumps(parsed)
                                                            fix_details["before_after_examples"].append({
                                                                "before": row,
                                                                "after": fixed_row,
                                                                "explanation": f"Extracted and fixed partial JSON from '{row[col]}'"
                                                            })
                                                            break
                                                        except:
                                                            pass
                                                
                                                # Try parsing key-value pairs
                                                parts = [p.strip() for p in val.split(',')]
                                                for part in parts:
                                                    if ':' in part:
                                                        k, v = part.split(':', 1)
                                                        k = k.strip().strip('"{}').strip()
                                                        v = v.strip().strip('"{}').strip()
                                                        
                                                        # Try to convert to appropriate type
                                                        if v.lower() == 'true':
                                                            new_obj[k] = True
                                                        elif v.lower() == 'false':
                                                            new_obj[k] = False
                                                        elif v.isdigit():
                                                            new_obj[k] = int(v)
                                                        elif v.replace('.', '', 1).isdigit():
                                                            new_obj[k] = float(v)
                                                        else:
                                                            new_obj[k] = v
                                            except:
                                                pass
                                        else:
                                            # For plain text, try to intelligently map to schema
                                            if 'notes' in json_schema:
                                                new_obj['notes'] = str(val)
                                            elif 'description' in json_schema:
                                                new_obj['description'] = str(val)
                                            elif 'text' in json_schema:
                                                new_obj['text'] = str(val)
                                            elif 'location' in json_schema and len(str(val).split()) <= 3:
                                                new_obj['location'] = str(val)
                                            else:
                                                string_keys = [k for k, v in json_schema.items() if isinstance(v, str)]
                                                if string_keys:
                                                    new_obj[string_keys[0]] = str(val)
                                                else:
                                                    first_key = list(json_schema.keys())[0] if json_schema else 'notes'
                                                    new_obj[first_key] = str(val)
                                        
                                        fixed_row[col] = json.dumps(new_obj)
                                        fix_details["before_after_examples"].append({
                                            "before": row,
                                            "after": fixed_row,
                                            "explanation": f"Invalid JSON '{row[col]}' was converted to valid JSON format matching the inferred schema: {json_schema}"
                                        })
                                        break
                        
                        elif issue.get("type") == "out_of_range" and issue.get("location") == "PriorityLevel":
                            # Find example of out-of-range PriorityLevel
                            invalid_rows = df[~df['PriorityLevel'].between(1, 5)].head(1)
                            if not invalid_rows.empty:
                                row = invalid_rows.iloc[0].to_dict()
                                fixed_row = row.copy()
                                original_value = row["PriorityLevel"]
                                fixed_row["PriorityLevel"] = min(max(row["PriorityLevel"], 1), 5)
                                fix_details["before_after_examples"].append({
                                    "before": row,
                                    "after": fixed_row,
                                    "explanation": f"Out-of-range PriorityLevel {original_value} was corrected to {fixed_row['PriorityLevel']}"
                                })
                        
                        elif issue.get("type") == "invalid_task_ids" and issue.get("location") == "RequestedTaskIDs":
                            # Find example of invalid TaskIDs
                            def has_invalid_task_id(task_list):
                                if not isinstance(task_list, str):
                                    return False
                                tasks = task_list.split(',')
                                for task in tasks:
                                    task = task.strip()
                                    if not task:
                                        continue
                                    if not (task.startswith('T') and task[1:].isdigit()):
                                        return True
                                    elif task.startswith('T') and task[1:].isdigit() and int(task[1:]) > 50:
                                        return True
                                return False
                            
                            invalid_rows = df[df['RequestedTaskIDs'].apply(has_invalid_task_id)].head(1)
                            if not invalid_rows.empty:
                                row = invalid_rows.iloc[0].to_dict()
                                fixed_row = row.copy()
                                original_value = row["RequestedTaskIDs"]
                                
                                tasks = original_value.split(',')
                                valid_tasks = [t.strip() for t in tasks 
                                               if t.strip().startswith('T') and 
                                               t.strip()[1:].isdigit() and 
                                               1 <= int(t.strip()[1:]) <= 50]
                                fixed_row["RequestedTaskIDs"] = ','.join(valid_tasks)
                                
                                fix_details["before_after_examples"].append({
                                    "before": row,
                                    "after": fixed_row,
                                    "explanation": f"Invalid TaskIDs '{original_value}' were corrected to '{fixed_row['RequestedTaskIDs']}'"
                                })
                        
                        elif issue.get("type") == "missing_values" and issue.get("location"):
                            # Find example of missing values
                            col = issue.get("location")
                            if col in df.columns:
                                missing_rows = df[df[col].isnull()].head(1)
                                if not missing_rows.empty:
                                    row = missing_rows.iloc[0].to_dict()
                                    fixed_row = row.copy()
                                    
                                    # Determine appropriate default value
                                    default_value = 0
                                    if "name" in col.lower() or "text" in col.lower() or "description" in col.lower():
                                        default_value = "Unknown"
                                    elif "date" in col.lower():
                                        default_value = pd.Timestamp('today')
                                    elif "json" in col.lower() or "JSON" in col:
                                        default_value = "{}"
                                    
                                    fixed_row[col] = default_value
                                    fix_details["before_after_examples"].append({
                                        "before": row,
                                        "after": fixed_row,
                                        "explanation": f"Missing value in {col} was filled with default value: {default_value}"
                                    })
                    except Exception as e:
                        print(f"Error generating before/after example for {issue.get('type')}: {str(e)}")
                    
                    # Add fix to appropriate category
                    if issue.get("severity") == "critical":
                        critical_fixes[issue_id] = fix_details
                    else:
                        warning_fixes[issue_id] = fix_details
                
                # Combine fixes with critical issues first
                result["fixes"] = {
                    "critical": critical_fixes,
                    "warnings": warning_fixes
                }
            
            print(f"Analysis complete. Found {len(all_issues)} issues.")
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing file: {str(e)}",
                "details": {
                    "error_type": type(e).__name__,
                    "file_path": file_path
                }
            }
            
    def fix_file_errors(self, file_path: str, output_path: str = None, fix_ids: List[int] = None) -> dict:
        """
        Fixes errors in a CSV or Excel file based on analysis results.
        
        Args:
            file_path: Path to the input file
            output_path: Path for the fixed output file (if None, will append '_fixed' to original name)
            fix_ids: List of issue IDs to fix (if None, will fix all issues)
            
        Returns:
            A dictionary with the fix results and path to the fixed file
        """
        # Log the start of the operation
        print(f"Starting fix operation for file: {file_path}")
        
        # Validate input parameters
        if not file_path:
            return {
                "status": "error",
                "message": "No file path provided"
            }
            
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}"
            }
            
        # Validate file type
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in ['.csv', '.xlsx', '.xls']:
            return {
                "status": "error",
                "message": f"Unsupported file type: {file_ext}. Please provide a CSV or Excel file."
            }

        # First analyze the file
        analysis = self.analyze_file(file_path)
        
        if analysis["status"] == "error":
            return analysis
            
        # Check if there are issues to fix
        if "detailed_analysis" not in analysis or "issues" not in analysis["detailed_analysis"]:
            return {
                "status": "success",
                "message": "No issues found that require fixing",
                "fixed_file": None
            }
            
        # Set default output path if not provided
        if not output_path:
            file_name = os.path.basename(file_path)
            file_dir = os.path.dirname(file_path)
            file_base, file_ext = os.path.splitext(file_name)
            output_path = os.path.join(file_dir, f"{file_base}_fixed{file_ext}")
        
        print(f"Output will be saved to: {output_path}")
        
        # Validate output path
        try:
            # Check if output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created output directory: {output_dir}")
                
            # Check if output path is writable by trying to open it
            with open(output_path, 'a') as f:
                pass
            os.remove(output_path)  # Remove the test file
            print("Output path is valid and writable")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cannot write to output path: {output_path}. Error: {str(e)}"
            }
            
        # Load the file
        try:
            if file_ext == '.csv':
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='latin1')
                    except:
                        df = pd.read_csv(file_path, encoding='cp1252')
                except pd.errors.ParserError:
                    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported file type: {file_ext}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error reading file: {str(e)}"
            }
            
        # Get issues to fix
        all_issues = analysis["detailed_analysis"]["issues"]
        issues_to_fix = all_issues
        
        # Validate fix_ids if provided
        if fix_ids is not None:
            if not isinstance(fix_ids, list):
                try:
                    fix_ids = [int(fix_ids)]
                except:
                    return {
                        "status": "error",
                        "message": f"Invalid fix_ids: {fix_ids}. Must be a list of integers or convertible to integers."
                    }
            
            # Validate each fix_id
            valid_fix_ids = []
            for fix_id in fix_ids:
                try:
                    fix_id = int(fix_id)
                    if 0 <= fix_id < len(all_issues):
                        valid_fix_ids.append(fix_id)
                    else:
                        return {
                            "status": "error",
                            "message": f"Invalid fix_id: {fix_id}. Must be between 0 and {len(all_issues)-1}"
                        }
                except:
                    return {
                        "status": "error",
                        "message": f"Invalid fix_id: {fix_id}. Must be an integer."
                    }
                    
            fix_ids = valid_fix_ids
            if not fix_ids:
                return {
                    "status": "error",
                    "message": "No valid fix IDs provided"
                }
                
            issues_to_fix = [issue for i, issue in enumerate(all_issues) if i in fix_ids]
            
        # Apply fixes
        fixes_applied = []
        fixes_failed = []
        
        # Keep a backup of original dataframe for validation
        df_original = df.copy()
        
        print(f"Starting to apply {len(issues_to_fix)} fixes...")
        
        # Reorder issues to fix critical validation issues first
        # This ensures that fixes like PriorityLevel range are applied before duplicate ID fixes
        prioritized_issues = []
        medium_priority_issues = []
        low_priority_issues = []
        
        for issue in issues_to_fix:
            issue_type = issue.get("type", "unknown")
            issue_location = issue.get("location", "unknown")
            issue_severity = issue.get("severity", "warning")
            
            # First prioritize by severity
            if issue_severity == "critical":
                # Among critical issues, prioritize specific ones first
                if issue_type == "out_of_range" and issue_location == "PriorityLevel":
                    # Fix PriorityLevel range issues first since other fixes may depend on valid values
                    prioritized_issues.insert(0, issue)
                elif issue_type == "invalid_json":
                    # Fix JSON issues next
                    prioritized_issues.append(issue)
                elif issue_type == "invalid_task_ids":
                    # Then fix invalid TaskIDs
                    prioritized_issues.append(issue)
                elif issue_type == "duplicate_id":
                    # Defer duplicate ID fixes to the end of critical issues
                    medium_priority_issues.append(issue)
                else:
                    # Other critical issues
                    prioritized_issues.append(issue)
            else:
                # All warnings go to low priority
                low_priority_issues.append(issue)
                
        # Combine the lists in order of priority
        ordered_issues = prioritized_issues + medium_priority_issues + low_priority_issues
        
        for i, issue in enumerate(ordered_issues):
            issue_type = issue.get("type", "unknown")
            issue_location = issue.get("location", "unknown")
            print(f"Processing fix {i+1}/{len(ordered_issues)}: {issue_type} in {issue_location}")
            
            try:
                # Get the fix code
                fix_code = issue.get("fix_code", "")
                if not fix_code:
                    print(f"- Failed: No fix code available")
                    fixes_failed.append({
                        "issue_id": i,
                        "reason": "No fix code available",
                        "issue": issue
                    })
                    continue
                
                # Validate fix code (security checks)
                if "import os" in fix_code or "import sys" in fix_code or "exec(" in fix_code or "eval(" in fix_code or "__" in fix_code:
                    print(f"- Failed: Fix code contains potentially unsafe operations")
                    # Check if we have a custom implementation for this issue type
                    if issue.get("type") == "invalid_json" and "JSON" in issue.get("location", ""):
                        print(f"  Using built-in implementation for invalid JSON fix")
                        has_custom_implementation = True
                    else:
                        fixes_failed.append({
                            "issue_id": i,
                            "reason": "Fix code contains potentially unsafe operations",
                            "issue": issue
                        })
                        continue
                    
                # Execute the fix code in a controlled environment
                try:
                    # Create a copy of the dataframe for this fix
                    df_temp = df.copy()
                    
                    # If the issue type is formatting, let's handle it with custom code
                    if issue.get("type") == "formatting" and issue.get("location") == "RequestedTaskIDs":
                        print("  - Using custom code for RequestedTaskIDs formatting...")
                        
                        # This is a safer approach than using the LLM-generated code
                        def fix_task_id_format(task_list):
                            if not isinstance(task_list, str):
                                return ""
                            tasks = task_list.split(',')
                            formatted_tasks = []
                            for task in tasks:
                                task = task.strip()
                                if task.startswith('T') and task[1:].isdigit() and 1 <= int(task[1:]) <= 50:
                                    # Ensure consistent format without leading zeros
                                    formatted_tasks.append(f"T{int(task[1:])}")
                            return ','.join(formatted_tasks)
                        
                        df_temp['RequestedTaskIDs'] = df_temp['RequestedTaskIDs'].apply(fix_task_id_format)
                    else:
                        # Initialize variables used for all code paths
                        exec_globals = {
                            '__builtins__': __builtins__,
                            'pd': pd,
                            'json': json,
                            're': re,
                            'math': __import__('math'),
                            'np': __import__('numpy'),
                            'isna': lambda x: pd.isna(x),
                            'notnull': lambda x: pd.notnull(x),
                            'valid_tasks': [],  # Common variable used in fix code
                            'valid_groups': []  # Common variable used in fix code
                        }
                        exec_locals = {"df": df_temp}
                    
                    # Handle specific issue types directly instead of using exec
                    if issue.get("type") == "out_of_range" and issue.get("location") == "PriorityLevel":
                        print(f"- Using custom fix for PriorityLevel range issues")
                        # Custom implementation for PriorityLevel range fix
                        affected_rows = (~df_temp['PriorityLevel'].between(1, 5)).sum()
                        df_temp['PriorityLevel'] = df_temp['PriorityLevel'].apply(
                            lambda x: min(max(x, 1), 5) if pd.notnull(x) else 3
                        )
                        print(f"  Fixed {affected_rows} rows with out-of-range PriorityLevel values")
                        
                    elif issue.get("type") == "invalid_json" and "JSON" in issue.get("location", ""):
                        print(f"- Using custom fix for invalid JSON in {issue.get('location')}")
                        # Custom implementation for JSON fix
                        col = issue.get("location")
                        
                        # Count affected rows before fix
                        invalid_count = sum(~df_temp[col].apply(self._is_valid_json))
                        
                        # Infer schema from valid JSON in this column
                        json_schema = self._infer_json_schema(df_temp, col)
                        print(f"  Inferred JSON schema: {json_schema}")
                        
                        # Fix invalid JSON using the inferred schema
                        def fix_json_with_schema(value):
                            if pd.isna(value):
                                return json.dumps(json_schema)
                                
                            if not isinstance(value, str) or not self._is_valid_json(value):
                                # Create a copy of the schema
                                new_obj = json_schema.copy()
                                
                                # Try to extract key-value pairs if it looks like JSON
                                if isinstance(value, str):
                                    # Check for partial JSON or key-value format
                                    if (':' in value) or ('{' in value and '}' in value) or ('"' in value):
                                        try:
                                            # For values that look like JSON with missing brackets
                                            if value.strip().startswith('"') and not value.strip().startswith('{"'):
                                                value = '{' + value + '}'
                                            
                                            # For values with malformed JSON
                                            if '{' in value and '}' in value:
                                                # Extract content between { and }
                                                json_content = re.search(r'\{(.*?)\}', value)
                                                if json_content:
                                                    extracted = '{' + json_content.group(1) + '}'
                                                    try:
                                                        parsed = json.loads(extracted)
                                                        return json.dumps(parsed)
                                                    except:
                                                        pass
                                                
                                                # Try parsing key-value pairs
                                                parts = [p.strip() for p in value.split(',')]
                                                for part in parts:
                                                    if ':' in part:
                                                        k, v = part.split(':', 1)
                                                        k = k.strip().strip('"{}').strip()
                                                        v = v.strip().strip('"{}').strip()
                                                        
                                                        # Try to convert to appropriate type
                                                        if v.lower() == 'true':
                                                            new_obj[k] = True
                                                        elif v.lower() == 'false':
                                                            new_obj[k] = False
                                                        elif v.isdigit():
                                                            new_obj[k] = int(v)
                                                        elif v.replace('.', '', 1).isdigit():
                                                            new_obj[k] = float(v)
                                                        else:
                                                            new_obj[k] = v
                                        except:
                                            # If key-value extraction fails, use plain text approach
                                            pass
                                    else:
                                        # For plain text, try to map to common schema fields
                                        # First look for 'notes' field
                                        if 'notes' in json_schema:
                                            new_obj['notes'] = str(value)
                                        elif 'description' in json_schema:
                                            new_obj['description'] = str(value)
                                        elif 'text' in json_schema:
                                            new_obj['text'] = str(value)
                                        # Then try using location for short text that could be a place
                                        elif 'location' in json_schema and len(str(value).split()) <= 3:
                                            new_obj['location'] = str(value)
                                        # Fallback to first string field
                                        else:
                                            string_keys = [k for k, v in json_schema.items() if isinstance(v, str)]
                                            if string_keys:
                                                new_obj[string_keys[0]] = str(value)
                                            else:
                                                # Last resort fallback
                                                first_key = list(json_schema.keys())[0] if json_schema else 'notes'
                                                new_obj[first_key] = str(value)
                                
                                return json.dumps(new_obj)
                            
                            return value
                        
                        df_temp[col] = df_temp[col].apply(fix_json_with_schema)
                        
                        print(f"  Fixed {invalid_count} rows with invalid JSON in {col}")
                        
                    elif issue.get("type") == "duplicate_id" and "ID" in issue.get("location", ""):
                        print(f"- Using custom fix for duplicate {issue.get('location')} values")
                        # Custom implementation for duplicate ID fix
                        col = issue.get("location")
                        
                        duplicate_indices = df_temp[df_temp[col].duplicated(keep='first')].index
                        duplicate_count = len(duplicate_indices)
                        
                        for idx in duplicate_indices:
                            df_temp.loc[idx, col] = f"{df_temp.loc[idx, col]}_dup"
                            
                        print(f"  Fixed {duplicate_count} duplicate {col} values")
                        
                    elif issue.get("type") == "invalid_task_ids" and issue.get("location") == "RequestedTaskIDs":
                        print(f"- Using custom fix for invalid TaskIDs")
                        # Custom implementation for TaskIDs fix
                        
                        # Define helper function for validation
                        def has_invalid_task_id(task_list):
                            if not isinstance(task_list, str):
                                return False
                            tasks = task_list.split(',')
                            for task in tasks:
                                task = task.strip()
                                if not (task.startswith('T') and task[1:].isdigit() and 1 <= int(task[1:]) <= 50):
                                    return True
                            return False
                        
                        # Count affected rows before fix
                        invalid_count = df_temp['RequestedTaskIDs'].apply(has_invalid_task_id).sum()
                        
                        df_temp['RequestedTaskIDs'] = df_temp['RequestedTaskIDs'].apply(
                            lambda task_list: ','.join([t.strip() for t in task_list.split(',') 
                                                      if isinstance(task_list, str) and 
                                                      t.strip().startswith('T') and 
                                                      t.strip()[1:].isdigit() and 
                                                      1 <= int(t.strip()[1:]) <= 50]) 
                                              if isinstance(task_list, str) else ''
                        )
                        
                        print(f"  Fixed {invalid_count} rows with invalid TaskIDs")
                        
                    elif issue.get("type") == "inconsistent_text_case" and issue.get("location") == "GroupTag":
                        print(f"- Using custom fix for inconsistent text case in GroupTag")
                        # Custom implementation for text case standardization
                        # Standardize to title case (e.g., "GroupA")
                        
                        # Count affected rows before fix
                        original_values = set(df_temp['GroupTag'].unique())
                        df_temp['GroupTag'] = df_temp['GroupTag'].apply(
                            lambda x: x.title() if isinstance(x, str) else x
                        )
                        
                        # Count the fixed rows
                        standardized_values = set(df_temp['GroupTag'].unique())
                        fixed_count = len(original_values) - len(standardized_values)
                        print(f"  Standardized {fixed_count} inconsistent GroupTag values")
                        
                    elif issue.get("type") == "leading_trailing_whitespace" and issue.get("location") == "ClientName":
                        print(f"- Using custom fix for whitespace issues in ClientName")
                        # Custom implementation for whitespace trimming
                        
                        # Count affected rows before fix
                        whitespace_count = sum(df_temp['ClientName'].apply(
                            lambda x: isinstance(x, str) and (x.strip() != x)
                        ))
                        
                        df_temp['ClientName'] = df_temp['ClientName'].apply(
                            lambda x: x.strip() if isinstance(x, str) else x
                        )
                        
                        print(f"  Trimmed whitespace from {whitespace_count} ClientName values")
                        
                    else:
                        print(f"- Using general fix code execution")
                        # For other cases, try to execute the fix code with proper environment
                       
                        # Prepare helper functions
                        helper_functions = """
def isna(val):
    import math
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False

def notnull(val):
    return not isna(val)
"""
                        
                        # Add helper functions and necessary imports
                        complete_fix_code = f"""
import pandas as pd
import numpy as np
import json
import re
import math

{helper_functions}

# Fix code starts here
{fix_code.replace("pd.isna", "isna").replace("pd.isnull", "isna").replace("pd.notnull", "notnull")}
"""
                        
                        # Create a clean execution environment with necessary imports
                        exec_globals = {
                            '__builtins__': __builtins__,
                            'pd': pd,
                            'json': json,
                            're': re,
                            'math': __import__('math'),
                            'np': __import__('numpy'),
                            'isna': lambda x: pd.isna(x),
                            'notnull': lambda x: pd.notnull(x),
                            'valid_tasks': [],  # Common variable used in fix code
                            'valid_groups': []  # Common variable used in fix code
                        }
                        exec_locals = {"df": df_temp}
                        
                        # Execute the fix code
                        exec(complete_fix_code, exec_globals, exec_locals)
                        
                        # Get the updated dataframe
                        if "df" in exec_locals and isinstance(exec_locals["df"], pd.DataFrame):
                            df_temp = exec_locals["df"]
                        else:
                            raise ValueError("Fix code did not properly update the dataframe")
                    
                    # Verify the fix didn't drastically change the dataframe size
                    if len(df_temp) < len(df) * 0.9:  # Allow for some row reduction, but not too much
                        print(f"- Failed: Fix would remove too many rows (from {len(df)} to {len(df_temp)})")
                        fixes_failed.append({
                            "issue_id": i,
                            "reason": f"Fix would remove too many rows (from {len(df)} to {len(df_temp)})",
                            "issue": issue
                        })
                        continue
                    
                    # Verify columns haven't been drastically changed
                    removed_cols = set(df.columns) - set(df_temp.columns)
                    if removed_cols:
                        print(f"- Failed: Fix would remove columns: {', '.join(removed_cols)}")
                        fixes_failed.append({
                            "issue_id": i,
                            "reason": f"Fix would remove columns: {', '.join(removed_cols)}",
                            "issue": issue
                        })
                        continue
                    
                    # Check for any new errors introduced by the fix
                    try:
                        # Get a sample of the data to validate
                        sample_data = df_temp.head(min(5, len(df_temp))).to_dict(orient='records')
                        
                        # Simple validation of specific datatypes based on column names
                        validation_issues = []
                        
                        # Check ID columns
                        for col in df_temp.columns:
                            if col.endswith('ID') and 'PriorityLevel' in df_temp.columns:
                                # Check that IDs are not numeric in columns like ClientID
                                invalid_ids = df_temp[pd.to_numeric(df_temp[col], errors='coerce').notnull()].shape[0]
                                if invalid_ids > 0:
                                    validation_issues.append(f"{invalid_ids} rows have numeric {col} values")
                            
                            # Check PriorityLevel range
                            elif col == 'PriorityLevel' and df_temp[col].dtype in ['int64', 'float64']:
                                invalid_priority = df_temp[~df_temp[col].between(1, 5)].shape[0]
                                if invalid_priority > 0:
                                    validation_issues.append(f"{invalid_priority} rows still have PriorityLevel outside valid range (1-5)")
                        
                        if validation_issues:
                            print(f"- Failed: Fix introduced new validation issues: {'; '.join(validation_issues)}")
                            fixes_failed.append({
                                "issue_id": i,
                                "reason": f"Fix introduced new validation issues: {'; '.join(validation_issues)}",
                                "issue": issue
                            })
                            continue
                    except Exception as validation_error:
                        print(f"- Warning: Error during validation checks: {str(validation_error)}")
                        # Continue with the fix even if validation fails
                        
                    # If all validations pass, apply the fix to the main dataframe
                    df = df_temp
                    print(f"- Success: Applied fix for {issue_type} in {issue_location}")
                    
                    fixes_applied.append({
                        "issue_id": i,
                        "issue_type": issue.get("type"),
                        "location": issue.get("location"),
                        "description": issue.get("description"),
                        "fix": issue.get("fix")
                    })
                    
                except Exception as e:
                    print(f"- Failed: Error executing fix code: {str(e)}")
                    fixes_failed.append({
                        "issue_id": i,
                        "reason": f"Error executing fix code: {str(e)}",
                        "issue": issue
                    })
                    
            except Exception as e:
                print(f"- Failed: Unexpected error: {str(e)}")
                fixes_failed.append({
                    "issue_id": i,
                    "reason": str(e),
                    "issue": issue
                })
                
        # If no fixes were applied, return the original dataframe
        if not fixes_applied:
            print("No fixes were applied successfully.")
            return {
                "status": "warning",
                "message": "No fixes were applied",
                "fixes_failed": fixes_failed,
                "fixed_file": None
            }
                
        # Save the fixed file
        try:
            print(f"Saving fixed file to {output_path}")
            
            # Compare before and after metrics
            metrics = {
                "before": {
                    "rows": len(df_original),
                    "columns": len(df_original.columns),
                    "null_values": df_original.isnull().sum().sum(),
                    "duplicate_rows": df_original.duplicated().sum()
                },
                "after": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "null_values": df.isnull().sum().sum(),
                    "duplicate_rows": df.duplicated().sum()
                }
            }
            
            # Add specific metrics for known issue types
            if 'PriorityLevel' in df.columns and df['PriorityLevel'].dtype in ['int64', 'float64']:
                metrics["before"]["invalid_priority"] = df_original[~df_original['PriorityLevel'].between(1, 5)].shape[0]
                metrics["after"]["invalid_priority"] = df[~df['PriorityLevel'].between(1, 5)].shape[0]
                
            # Calculate success rate
            total_issues = len(issues_to_fix)
            successful_fixes = len(fixes_applied)
            success_rate = (successful_fixes / total_issues) * 100 if total_issues > 0 else 0
            
            # Save the file
            if file_ext == '.csv':
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)
                
            # Count fixes by severity
            critical_fixes_applied = [fix for fix in fixes_applied if fix.get("issue", {}).get("severity") == "critical"]
            warning_fixes_applied = [fix for fix in fixes_applied if fix.get("issue", {}).get("severity") != "critical"]
            
            # Count total issues by severity
            critical_issues_count = len([issue for issue in issues_to_fix if issue.get("severity") == "critical"])
            warning_issues_count = len([issue for issue in issues_to_fix if issue.get("severity") != "critical"])
            
            # Calculate success rate with weighted importance
            critical_success_rate = len(critical_fixes_applied) / critical_issues_count if critical_issues_count > 0 else 1.0
            warning_success_rate = len(warning_fixes_applied) / warning_issues_count if warning_issues_count > 0 else 1.0
            
            # Print success message with severity breakdown
            success_msg = f"File saved successfully. Fixed {successful_fixes} out of {total_issues} issues ({success_rate:.1f}%)."
            if critical_issues_count > 0:
                success_msg += f" Critical issues: {len(critical_fixes_applied)}/{critical_issues_count} fixed."
            if warning_issues_count > 0:
                success_msg += f" Warnings: {len(warning_fixes_applied)}/{warning_issues_count} fixed."
            
            print(success_msg)
                
            result = {
                "status": "success",
                "message": f"Fixed {len(fixes_applied)} issues out of {len(issues_to_fix)} ({success_rate:.1f}%)",
                "critical_fixes": {
                    "applied": len(critical_fixes_applied),
                    "total": critical_issues_count
                },
                "warning_fixes": {
                    "applied": len(warning_fixes_applied),
                    "total": warning_issues_count
                },
                "fixed_file": output_path,
                "fixes_applied": fixes_applied,
                "fixes_failed": fixes_failed,
                "metrics": metrics
            }
            
            return result
            
        except Exception as e:
            print(f"Error saving fixed file: {str(e)}")
            return {
                "status": "error",
                "message": f"Error saving fixed file: {str(e)}",
                "fixes_applied": fixes_applied,
                "fixes_failed": fixes_failed
            }
    
    def _is_valid_json(self, text):
        """
        Helper method to check if a string is valid JSON.
        Handles various edge cases and returns False for non-string inputs.
        """
        if not isinstance(text, str):
            return False
            
        # Empty strings are not valid JSON
        if not text.strip():
            return False
            
        try:
            json.loads(text)
            return True
        except Exception:
            # Check if it might be a malformed JSON with missing braces
            if text.strip().startswith('"') and text.strip().endswith('"'):
                try:
                    json.loads('{' + text + '}')
                    return False  # It's still invalid as standalone JSON
                except:
                    pass
            return False
    
    def _validate_model(self, model_name: str) -> bool:
        """
        Validates if the given model name is supported.
        Returns True if valid, False otherwise.
        """
        try:
            # Check if model exists in available models list
            if model_name in GITHUB_AI_MODELS.values():
                return True
                
            # Check if it matches pattern for different providers
            if (model_name.startswith("openai/") or 
                model_name.startswith("meta/") or 
                model_name.startswith("meta-llama/") or 
                model_name.startswith("anthropic/") or
                model_name.startswith("deepseek/")):
                return True
                
            return False
        except Exception:
            return False

    def _infer_json_schema(self, df, column):
        """
        Infers the JSON schema from valid JSON rows in a column.
        Returns a dictionary of most common keys and their value types.
        """
        schema = {}
        valid_jsons = []
        
        # First pass: collect all valid JSON objects
        for val in df[column]:
            if isinstance(val, str) and self._is_valid_json(val):
                try:
                    json_obj = json.loads(val)
                    if isinstance(json_obj, dict):
                        valid_jsons.append(json_obj)
                except:
                    pass
        
        if not valid_jsons:
            # No valid JSON found, return a default schema
            return {"notes": ""}
        
        # Collect keys and their frequencies
        key_freq = {}
        key_types = {}
        
        for json_obj in valid_jsons:
            for key, value in json_obj.items():
                key_freq[key] = key_freq.get(key, 0) + 1
                
                # Track the types of values for each key
                value_type = type(value).__name__
                if key not in key_types:
                    key_types[key] = {value_type: 1}
                else:
                    key_types[key][value_type] = key_types[key].get(value_type, 0) + 1
        
        # For each key, determine most frequent type
        for key, type_counts in key_types.items():
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            
            # Create default values based on type
            if most_common_type == 'str':
                schema[key] = ""
            elif most_common_type == 'int':
                schema[key] = 0
            elif most_common_type == 'float':
                schema[key] = 0.0
            elif most_common_type == 'bool':
                schema[key] = False
            elif most_common_type == 'dict':
                schema[key] = {}
            elif most_common_type == 'list':
                schema[key] = []
        
        # Sort keys by frequency (most common first)
        sorted_keys = sorted(key_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Create schema with most common keys
        final_schema = {}
        for key, _ in sorted_keys:
            final_schema[key] = schema[key]
            
        return final_schema

    def _determine_issue_severity(self, issue_type, count=None, impact=None):
        """
        Determines if an issue is critical or just a warning based on its type and attributes.
        
        Args:
            issue_type: The type of the issue
            count: The number of affected rows/elements (if available)
            impact: The impact description (if available)
            
        Returns:
            'critical' or 'warning' based on issue severity
        """
        # Define critical issues - these require immediate attention
        critical_issue_types = [
            "duplicate_id",           # Duplicate IDs break data integrity 
            "invalid_json",           # Invalid JSON can break parsing
            "out_of_range",           # Out of range values violate business rules
            "invalid_task_ids",       # Invalid task IDs can break references
            "null_in_required_field", # Missing required data
            "type_mismatch",          # Incompatible data types
            "business_rule_violation" # Violates core business rules
        ]
        
        # Issues that are critical if they occur frequently
        frequency_dependent_issues = [
            "missing_values",         # Missing values might be critical if widespread
            "duplicate_rows"          # Duplicate rows might be critical if numerous
        ]
        
        # Determine severity
        if issue_type in critical_issue_types:
            return "critical"
        elif issue_type in frequency_dependent_issues and count and count > 5:
            # If more than 5 instances, consider it critical
            return "critical"
        elif impact and any(term in impact.lower() for term in 
                          ["severe", "critical", "break", "corrupt", "fail", "crash"]):
            # Impact-based determination
            return "critical"
        else:
            # Everything else is a warning
            return "warning"
