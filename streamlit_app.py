import streamlit as st
import requests
import pandas as pd
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Data Alchemist", layout="wide")
st.title("�‍♂️ Data Alchemist - AI Data Validation & Correction")

# Initialize session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Mode switch: text or file
mode = st.sidebar.radio("Choose Input Mode", ["Text Input", "File Upload"])

# ------------------------------
# TEXT MODE: For all AI features
# ------------------------------
if mode == "Text Input":
    st.sidebar.title("Choose AI Feature")
    option = st.sidebar.selectbox("AI Feature", [
        "Header Mapping",
        "Row Validation",
        "Natural Language Search",
        "Natural Language Modification",
        "Suggest Row Correction",
        "Rule Generator",
        "Rule Recommendations",
        "Priority Profile Generator"
    ])

    input_text = st.text_area("✍️ Enter text input below", height=150)

    if st.button("🚀 Run AI"):
        if not input_text:
            st.warning("Please enter some text.")
        else:
            with st.spinner("Thinking..."):
                endpoint = {
                    "Header Mapping": "/ai-header-map",
                    "Row Validation": "/ai-validate",
                    "Natural Language Search": "/ai-nl-search",
                    "Natural Language Modification": "/ai-nl-modify",
                    "Suggest Row Correction": "/ai-correct",
                    "Rule Generator": "/ai-rule-gen",
                    "Rule Recommendations": "/ai-rule-hints",
                    "Priority Profile Generator": "/ai-priority"
                }[option]

                try:
                    res = requests.post(f"{API_URL}{endpoint}", json={"text": input_text})
                    res.raise_for_status()
                    st.success("✅ AI Response")
                    st.json(res.json())
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ------------------------------
# FILE MODE: Upload CSV/XLSX
# ------------------------------
elif mode == "File Upload":
    st.subheader("📤 Upload a CSV or Excel File")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

    if uploaded_file and st.button("� Analyze File"):
        with st.spinner("Analyzing file..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                res = requests.post(f"{API_URL}/upload-data/", files=files)
                res.raise_for_status()
                result = res.json()

                # Store the analysis result in session state for later use
                st.session_state.analysis_result = result
                
                st.success("✅ File Analysis Complete")
                
                # Show which analysis method was used
                if not result.get("using_gpt", False):
                    st.warning("⚠️ Using limited local analysis (GPT AI not available)")
                else:
                    st.success("✅ Using advanced AI analysis")
                
                # Display file preview
                st.subheader("📊 File Preview (Top 5 Rows)")
                st.json(result.get("preview"))
                
                # Display analysis results
                analysis = result.get("analysis", {})
                
                if analysis.get("status") == "success":
                    summary = analysis.get("summary", {})
                    total_issues = summary.get("total_issues", 0)
                    
                    # Display file statistics
                    st.subheader("📋 File Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    file_info = analysis.get("file_info", {})
                    
                    with col1:
                        st.metric("Rows", file_info.get("rows", "N/A"))
                    with col2:
                        st.metric("Columns", file_info.get("columns", "N/A"))
                    with col3:
                        st.metric("File Type", file_info.get("file_type", "N/A"))
                    with col4:
                        st.metric("File Size", file_info.get("file_size", "N/A"))
                    
                    # Display issues summary
                    st.subheader("🔍 Issues Summary")
                    if total_issues == 0:
                        st.success("✅ No issues found in the file!")
                    else:
                        st.error(f"❌ Found {total_issues} issues in the file")
                        
                        # Show summary by issue type
                        issue_types = {}
                        if "detailed_analysis" in analysis and "issues" in analysis["detailed_analysis"]:
                            for issue in analysis["detailed_analysis"]["issues"]:
                                issue_type = issue.get("type", "unknown")
                                if issue_type in issue_types:
                                    issue_types[issue_type] += 1
                                else:
                                    issue_types[issue_type] = 1
                                    
                        # Display issue type counts
                        st.subheader("Issues by Type")
                        for issue_type, count in issue_types.items():
                            st.metric(issue_type.replace("_", " ").title(), count)
                        
                        # Display detailed issues with fix recommendations
                        st.subheader("🛠️ Detailed Issues & Fixes")
                        
                        fixes = analysis.get("fixes", {})
                        for issue_id, fix_details in fixes.items():
                            with st.expander(f"{fix_details.get('type', 'Issue').replace('_', ' ').title()} - {fix_details.get('description', 'No description')}"):
                                st.markdown(f"**Location:** {fix_details.get('location', 'Unknown')}")
                                st.markdown(f"**Impact:** {fix_details.get('impact', 'Unknown')}")
                                st.markdown(f"**Fix Recommendation:** {fix_details.get('fix_recommendation', 'No recommendation')}")
                                
                                # Show before/after examples if available
                                if fix_details.get("before_after_examples"):
                                    st.subheader("Example Fix")
                                    for example in fix_details["before_after_examples"]:
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.markdown("**Before:**")
                                            st.json(example.get("before", {}))
                                        with col2:
                                            st.markdown("**After:**")
                                            st.json(example.get("after", {}))
                        
                        # Show fix button if fixes are available
                        if result.get("fix_info", {}).get("available", False):
                            st.subheader("🔧 Fix Issues")
                            if st.button("Fix All Issues"):
                                with st.spinner("Applying fixes..."):
                                    fix_response = requests.post(
                                        f"{API_URL}/fix-file/", 
                                        json={"file_path": result.get("temp_file_path")}
                                    )
                                    
                                    if fix_response.status_code == 200:
                                        fix_result = fix_response.json()
                                        
                                        if "error" in fix_result:
                                            st.error(fix_result["error"])
                                        else:
                                            st.success(f"✅ {fix_result['message']}")
                                            
                                            # Show download link
                                            download_url = fix_result.get("download_url")
                                            if download_url:
                                                st.markdown(f"[⬇️ Download Fixed File]({API_URL}{download_url})")
                                            
                                            # Show preview of fixed file
                                            st.subheader("Fixed File Preview")
                                            st.json(fix_result.get("fixed_preview", []))
                                            
                                            # Show details of fixes applied
                                            st.subheader("Fixes Applied")
                                            for i, fix in enumerate(fix_result.get("fixes_applied", []), 1):
                                                st.markdown(f"**Fix {i}:** {fix.get('issue_type', 'Unknown')} in {fix.get('location', 'Unknown')}")
                                    else:
                                        st.error(f"❌ Error fixing file: {fix_response.text}")
                else:
                    st.error(f"❌ Error analyzing file: {analysis.get('message', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error during file analysis: {e}")
                st.error("Please check if the backend server is running properly.")
