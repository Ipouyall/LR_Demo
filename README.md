# Literature Review Dashboard

![Overall Dashboard Demo](docs/overall_dashboard.webp)

A comprehensive Streamlit application designed for researchers and academics to explore, analyze, and synthesize academic literature. The platform offers both standard analytical tools and advanced AI-powered capabilities to accelerate the literature review process.

## Key Features

The application operates in two primary modes:

### Standard Mode
- **Knowledge Base Management**: Build and curate a personal collection of academic papers.
- **Advanced Filtering**: Navigate literature by year, author, and keywords.
- **Publication Analytics**: Visualize publication trends over time.
- **Keyword Analysis**: Generate word clouds to identify prevalent themes in your collection.
- **Literature Export**: Summarize and export findings for your research.

### AI Mode
*Includes all Standard Mode features, plus:*
- **Paper Chat**: Interact directly with specific papers to extract methodologies, results, and limitations.

  ![Paper Chat Demo](docs/chat_with_paper.webp)

- **Automated Summaries**: Generate comprehensive literature overviews and methodology comparisons across multiple papers.
- **Research Insights**: Conduct thematic analysis and discover citation suggestions.
- **Deep Research**: Utilize natural language queries to let the AI automatically extract keywords, search online databases (Semantic Scholar), deduplicate, and filter relevant papers.

  ![Deep Research Demo](docs/deep_research.webp)

## Session Analytics & Reporting
The application logs interactions and provides a comprehensive reporting view, useful for tracking review sessions or structured HCI studies.

![Final Analysis and Report](docs/final_analysis_and_report.webp)

## Prerequisites

- Python 3.9 or higher

API Keys are strictly stored in memory for the duration of the session and are not saved to disk.
- **Gemini API Key**: Required for AI Mode features. [Get a key](https://aistudio.google.com/apikey)
- **Semantic Scholar API Key**: Optional but recommended to increase search rate limits (from 100 to 5,000 requests per 5 minutes). [Get a key](https://www.semanticscholar.org/product/api)

## Installation and Setup

1. **Clone the repository** (if applicable) and navigate to the project directory:
   ```bash
   cd LR_Demo
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys (Optional)**:
   You can provide your API keys through the UI at runtime, or create a `.env` file in the project root for automatic loading:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key_here
   ```

## Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

Upon launching, the dashboard will prompt you to enter participant information (applicable for configured HCI studies) and select your preferred mode of operation.

## Architecture and Configuration

- `app.py`: Main Streamlit application and UI routing.
- `task_config.py`: Defines the tasks and evaluation criteria for specific study modes.
- `event_logger.py`: Handles session logging and metric computation.
