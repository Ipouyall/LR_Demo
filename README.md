# CS889 - Literature Review Dashboard

A Streamlit dashboard for exploring and analyzing academic papers.

## Requirements

- Python 3.9+

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## API Keys

### Gemini API (Required for AI Mode)

To use AI-powered features, you'll need a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

### Semantic Scholar API (Optional but Recommended)

For paper search features, a Semantic Scholar API key is optional but highly recommended. Without it, you're limited to 100 requests per 5 minutes. With an API key, you get 5,000 requests per 5 minutes.

Get your free API key at: [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)

### How to Provide API Keys

You can provide your API keys in two ways:

1. **Through the UI (Recommended)**: When you select "AI Mode", you'll be prompted to enter your API keys. The keys will be stored in memory for the current session only and will not be saved to disk.

2. **Via .env file**: Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
   ```
   This will automatically load the keys on startup.

## Features

- **Standard Mode**: Browse and analyze your paper collection
- **AI Mode**: Unlock AI-powered insights, summaries, and Q&A
- **Online Search**: Search Semantic Scholar for new papers
- **Analytics**: Visualize publication trends and keyword analysis
- **Deep Research**: AI-assisted literature discovery

