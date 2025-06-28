# Data Alchemist AI API

AI-powered spreadsheet validation, search, rule generation & correction using OpenAI GPT or GitHub AI.

## Setup

1. Create a `.env` file with your API credentials:

```
# GitHub Token for accessing GitHub's hosted OpenAI models
GITHUB_TOKEN=your_github_token_here

# Since there might be access issues with GitHub AI, let's also keep the OpenAI option
OPENAI_API_KEY=your_openai_api_key_here

# GitHub AI Inference Endpoint
GITHUB_AI_ENDPOINT=https://models.github.ai/inference
GITHUB_AI_MODEL=openai/gpt-4.1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up the Azure AI Inference SDK (for GitHub AI):

```bash
# On Windows
setup_azure_sdk.bat

# On Linux/Mac
pip install azure-ai-inference
python check_models.py
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

- AI-powered CSV header mapping
- Advanced data validation with GPT/GitHub AI
- Natural language search to JSON filter conversion
- Data correction suggestions
- Rule generation from natural language
- Priority profile generation

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

Run the model checker to diagnose any issues:

```bash
python check_models.py
```

If you're having issues with the GitHub AI models, try:
1. Ensuring your GitHub token has the right permissions
2. Using a standard OpenAI API key as a fallback
3. Checking network connectivity to the GitHub AI endpoint

## License

MIT
