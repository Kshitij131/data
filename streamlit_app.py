import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Data Alchemist Tester", layout="centered")
st.title("🧪 Data Alchemist - AI Feature Tester")

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

    if uploaded_file and st.button("🚀 Run File Validation"):
        with st.spinner("Uploading to backend..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                res = requests.post(f"{API_URL}/upload-data/", files=files)
                res.raise_for_status()
                result = res.json()

                st.success("✅ AI Processed File Successfully")
                st.subheader("📊 File Preview (Top 5 Rows)")
                st.json(result.get("preview"))

                # Display validation results
                st.subheader("🤖 AI Validation")
                validation_result = result.get("validation", {})
                errors = validation_result.get("errors", [])
                
                # Show which validation method was used
                if not result.get("using_gpt", False):
                    st.warning("⚠️ Using limited local validation (GPT AI not available)")
                else:
                    st.success("✅ Using GPT AI for advanced validation")
                
                if not errors:
                    st.success("✅ No validation issues found!")
                else:
                    st.error(f"❌ Found {len(errors)} validation issues:")
                    for i, error in enumerate(errors, 1):
                        error_type = error.get('type', 'unknown')
                        error_msg = error.get('message', 'No details available')
                        with st.expander(f"Issue #{i}: {error_type}"):
                            st.write(f"**Message:** {error_msg}")
                            if error.get('details'):
                                st.json(error.get('details'))

            except Exception as e:
                st.error(f"❌ Error during file upload: {e}")
