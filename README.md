# Data Alchemist AI API

AI-powered spreadsheet validation, search, rule generation & correction using GitHub AI via Azure AI Inference SDK.

## Setup

1. Create a `.env` file with your GitHub token credentials:

```
# GitHub Token for accessing GitHub's hosted AI models
# IMPORTANT: Your token MUST have 'models:read' permission
GITHUB_TOKEN=your_github_token_here

# GitHub AI Inference Endpoint and Model
GITHUB_AI_ENDPOINT=https://models.github.ai/inference
GITHUB_AI_MODEL=meta/Meta-Llama-3.1-70B-Instruct
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Test the setup:

```bash
# Test if your GitHub token and model access is working correctly
python test_project.py
```

## Running the Application

1. Start the API server:

```bash
uvicorn main:app --reload
```

2. In another terminal, start the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

3. Access the UI at http://localhost:8501

## Features

- AI-powered CSV/Excel file analysis and validation
- Automatic error detection with severity classification
- Intelligent fixes for common data issues:
  - Duplicate IDs
  - Invalid JSON
  - Out-of-range values
  - Invalid references
  - Inconsistent text formatting
- Natural language search to filter conversion
- Priority profile generation for scheduling tasks
- Rule generation from natural language

## AI Model Options

The application supports two different ways to access AI models:

1. **GitHub AI via Azure SDK** (recommended for GitHub Copilot users)
   - Requires a GitHub fine-grained token with GitHub AI model access
   - Uses the Azure AI Inference SDK for communication

2. **GitHub AI via OpenAI client** (alternative method)
   - Uses the OpenAI client with a custom base URL
   - May work in some cases where the Azure SDK fails

3. **Standard OpenAI** (fallback option)
   - Requires an OpenAI API key
   - Uses the standard OpenAI models

The application will automatically try all three methods in order and use the first one that works.

## Troubleshooting

If you're having issues with the GitHub AI models, try:
1. Ensuring your GitHub token has the right permissions (models:read)
2. Check network connectivity to the GitHub AI endpoint
3. Verify the model name is correct in your `.env` file

## Project Cleanup

To remove temporary and unnecessary files, run:

```bash
python cleanup.py
```

This will clean up test files, temporary files, and other non-essential files from the project.

## License

MIT
