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

# Available GitHub AI models
GITHUB_AI_MODELS = {
    "GPT_4": "openai/gpt-4",
    "GPT_4_TURBO": "openai/gpt-4-turbo",
    "GPT_4_1": "openai/gpt-4.1",
    "GPT_3_5_TURBO": "openai/gpt-3.5-turbo",
}

class GPTAgent:
    def __init__(self):
        # Initialize with GitHub AI
        github_token = os.getenv("GITHUB_TOKEN")
        
        if not github_token:
            raise ValueError("GitHub token not found. Please set GITHUB_TOKEN in your environment variables.")
            
        # Get GitHub AI endpoint and model from environment variables
        github_ai_endpoint = os.getenv("GITHUB_AI_ENDPOINT", "https://models.github.ai/inference")
        github_ai_model = os.getenv("GITHUB_AI_MODEL", "openai/gpt-4.1")
        
        # GitHub AI requires the token to have "models:read" permission
        # Make sure you have granted this permission to your token in GitHub settings
        print("⚠️ Note: Your GitHub token must have 'models:read' permission to access GitHub AI models")
        
        self.client = ChatCompletionsClient(
            endpoint=github_ai_endpoint,
            credential=AzureKeyCredential(github_token)
        )
        
        # Set model name - ensure it has the correct prefix
        if not github_ai_model.startswith("openai/") and not github_ai_model.startswith("meta-llama/"):
            github_ai_model = f"openai/{github_ai_model}"
            
        self.model_name = github_ai_model
        print(f"✅ Using GitHub AI model: {self.model_name} with Azure AI Inference SDK")
        
        # Load FAISS for header and row search
        self.header_faiss = FAISSSearcher("headers")
        self.row_faiss = FAISSSearcher("rows")

    def _generate_content(self, prompt: str) -> str:
        """Generate content using Azure AI Inference client with GitHub AI"""
        try:
            # Use Azure AI Inference client with GitHub AI configuration
            response = self.client.complete(
                messages=[
                    SystemMessage(content="You are a data validation and analysis assistant."),
                    UserMessage(content=prompt)
                ],
                model=self.model_name,
                temperature=0.1,  # Low temperature for more deterministic outputs
                max_tokens=2048   # Adjust as needed
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating content: {e}")
            # Simple fallback - return a message about the error
            return f"Error generating content. Please check your configuration and try again."

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
