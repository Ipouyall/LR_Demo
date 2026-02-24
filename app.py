"""
Literature Review Dashboard - A Streamlit application for exploring and analyzing academic papers.
"""

import json
import os
import time
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
from collections import Counter
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai
import requests

from event_logger import (
    log_event, init_session, generate_participant_id,
    load_participant_log, compute_derived_metrics, events_to_csv_string,
)

load_dotenv()

_gemini_client = None

def init_gemini():
    """Initialize Gemini API with key from session state or environment."""
    global _gemini_client
    api_key = st.session_state.get('runtime_api_key')
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if api_key and api_key != "your_api_key_here":
        _gemini_client = genai.Client(api_key=api_key)
        return True
    return False

def get_gemini_response(prompt: str, context: str = "") -> str:
    """Get response from Gemini API."""
    try:
        if not _gemini_client:
            init_gemini()
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        response = _gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def save_api_key_to_session(api_key: str):
    """Save Gemini API key to session state (runtime only, not persisted)."""
    global _gemini_client
    st.session_state['runtime_api_key'] = api_key
    _gemini_client = genai.Client(api_key=api_key)

def save_semantic_scholar_key_to_session(api_key: str):
    """Save Semantic Scholar API key to session state (runtime only, not persisted)."""
    st.session_state['runtime_ss_api_key'] = api_key

def get_semantic_scholar_key() -> str:
    """Get Semantic Scholar API key from session state or environment."""
    api_key = st.session_state.get('runtime_ss_api_key')
    if not api_key:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    return api_key

def validate_api_key(api_key: str) -> bool:
    """Validate API key by attempting to use Gemini."""
    try:
        client = genai.Client(api_key=api_key)
        client.models.generate_content(
            model='gemini-2.0-flash',
            contents="Hello"
        )
        return True
    except Exception:
        return False

st.set_page_config(
    page_title="Literature Review Dashboard",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,1,0" />
<style>
    /* Global styling - dark monochrome theme */
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 50%, #262626 100%);
    }

    /* Fix unreadable white text on yellow primary buttons */
    button[kind="primary"] {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }
    
    button[kind="primary"] p {
        color: #1a1a1a !important;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(90deg, #ffd700 0%, #ffb800 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(255,215,0,0.15);
        border-color: rgba(255,215,0,0.3);
    }
    
    .metric-value {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffd700;
    }
    
    .metric-icon {
        font-size: 2.5rem !important;
        color: rgba(255,215,0,0.8);
    }
    
    .metric-label {
        color: #a0a0a0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Paper card styling */
    .paper-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.01));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .paper-card:hover {
        border-color: rgba(255,215,0,0.4);
        box-shadow: 0 8px 32px rgba(255,215,0,0.1);
    }
    
    .paper-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #f0f0f0;
        margin-bottom: 0.5rem;
        line-height: 1.4;
    }
    
    .paper-authors {
        color: #ffd700;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    
    .paper-journal {
        color: #888888;
        font-size: 0.85rem;
        font-style: italic;
    }
    
    .paper-abstract {
        color: #e0e0e0;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 1rem;
        padding: 1rem;
        background: rgba(0,0,0,0.3);
        border-radius: 8px;
        border-left: 3px solid #ffd700;
    }
    
    /* Keyword tag styling */
    .keyword-tag {
        display: inline-block;
        background: rgba(255,215,0,0.15);
        color: #f0f0f0;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        border: 1px solid rgba(255,215,0,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #888888;
        background: transparent;
        padding: 0.5rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #ffd700, #ffb800);
        color: #0d0d0d !important;
        font-weight: 600;
    }
    
    /* Chart container */
    .chart-container {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Filter section styling */
    .filter-section {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Quick stats in filter */
    .quick-stat {
        display: inline-block;
        background: rgba(255,215,0,0.1);
        color: #ffd700;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        border: 1px solid rgba(255,215,0,0.2);
    }

    /* AI Tools Navigation Panel */
    .ai-nav-panel {
        background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .ai-nav-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #ffd700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(255,215,0,0.2);
    }

    .ai-tool-desc {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.5rem;
        line-height: 1.4;
    }

    /* Fixed Radio Button Layout for AI Tools */
    div.stRadio > div {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    div.stRadio > div > label {
        background: rgba(255,255,255,0.03);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.08);
        width: 100% !important;
        margin: 0;
        align-items: center;
        justify-content: flex-start;
    }
    div.stRadio > div > label:hover {
        background: rgba(255,255,255,0.08);
        border-color: rgba(255,215,0,0.3);
    }

    /* AI Content Panel */
    .ai-content-panel {
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        min-height: 400px;
    }

    .ai-content-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f0f0f0;
        margin-bottom: 0.25rem;
    }

    .ai-content-subtitle {
        font-size: 0.9rem;
        color: #a0a0a0;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path: str) -> list:
    """Load paper data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get('references', [])

def save_papers_to_json(papers: list):
    """Write the current papers list back to the JSON file."""
    data_path = Path(__file__).parent / "data" / "example-bib.json"
    with open(data_path, 'w') as f:
        json.dump({"references": papers}, f, indent=2)
    load_data.clear()

def add_paper_to_collection(paper: dict):
    """Add a paper from search results to the user's collection."""
    papers = st.session_state.get('user_papers', [])
    max_id = max((p.get('id', 0) for p in papers), default=0)
    paper_copy = dict(paper)
    paper_copy['id'] = max_id + 1
    paper_copy.pop('url', None)
    papers.append(paper_copy)
    st.session_state.user_papers = papers
    save_papers_to_json(papers)

def remove_paper_from_collection(paper_id):
    """Remove a paper by ID from the user's collection."""
    papers = st.session_state.get('user_papers', [])
    papers = [p for p in papers if p.get('id') != paper_id]
    st.session_state.user_papers = papers
    save_papers_to_json(papers)

def create_dataframe(papers: list) -> pd.DataFrame:
    """Convert papers list to DataFrame for analysis."""
    df_data = []
    for paper in papers:
        df_data.append({
            'id': paper.get('id'),
            'title': paper.get('title', 'Unknown'),
            'authors': ', '.join(paper.get('authors', [])),
            'author_count': len(paper.get('authors', [])),
            'year': paper.get('year'),
            'journal': paper.get('journal', 'Unknown'),
            'volume': paper.get('volume'),
            'issue': paper.get('issue'),
            'pages': paper.get('pages'),
            'doi': paper.get('doi'),
            'abstract': paper.get('abstract', ''),
            'keywords': paper.get('keywords', []),
            'keyword_count': len(paper.get('keywords', []))
        })
    return pd.DataFrame(df_data)

def render_metric_card(value, label, icon=""):
    """Render a styled metric card."""
    icon_html = f'<span class="material-symbols-rounded metric-icon">{icon}</span>' if icon else ''
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{icon_html}<span>{value}</span></div>
            <div class="metric-label">{label}</div>
        </div>
    """, unsafe_allow_html=True)

def render_paper_card(paper: dict):
    """Render a styled paper card."""
    authors = ', '.join(paper.get('authors', []))
    journal = paper.get('journal', 'Unknown')
    year = paper.get('year', '')
    volume = paper.get('volume', '')
    issue = paper.get('issue', '')
    pages = paper.get('pages', '')
    
    citation = f"{journal}"
    if volume:
        citation += f", Vol. {volume}"
    if issue:
        citation += f"({issue})"
    if pages:
        citation += f", pp. {pages}"
    if year:
        citation += f" ({year})"
    
    keywords_html = ""
    for kw in paper.get('keywords', []):
        keywords_html += f'<span class="keyword-tag">{kw}</span>'
    
    st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">{paper.get('title', 'Untitled')}</div>
            <div class="paper-authors">{authors}</div>
            <div class="paper-journal">{citation}</div>
            <div style="margin-top: 0.8rem;">{keywords_html}</div>
            <div class="paper-abstract">{paper.get('abstract', 'No abstract available.')}</div>
        </div>
    """, unsafe_allow_html=True)

def render_participant_setup():
    """Render participant setup page — first page shown in the session."""
    st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem 1rem;">
            <h1 class="main-header">Welcome to the Study</h1>
            <p class="sub-header" style="font-size: 1.2rem; max-width: 650px; margin: 0 auto 2rem auto;">
                Please provide your information below to begin the session.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("### Participant Information")

            # Propose a random ID if not already generated
            if "proposed_id" not in st.session_state:
                st.session_state.proposed_id = generate_participant_id()

            p_name = st.text_input(
                "Your Name",
                value=st.session_state.get("participant_name", ""),
                placeholder="e.g. Jane Doe",
                help="Used to personalise your experience",
            )
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                lit_exp_levels = {
                    1: "Novice (Never done before)",
                    2: "Beginner (Done it once or twice)",
                    3: "Intermediate (Do it occasionally)",
                    4: "Advanced (Do it regularly)",
                    5: "Expert (Extensive experience)"
                }
                
                lit_exp_val = st.slider(
                    "Previous experience with literature review",
                    min_value=1, max_value=5, value=st.session_state.get("lit_exp_val", 3),
                    key="lit_exp_slider"
                )
                st.markdown(f"<p style='color: #ffd700; font-size: 0.9em; margin-top: -10px;'>Experience level: <b>{lit_exp_levels[lit_exp_val]}</b></p>", unsafe_allow_html=True)
                
            with col_exp2:
                ai_exp_levels = {
                    1: "Novice (Never used AI in research)",
                    2: "Beginner (Used occasionally for basic tasks)",
                    3: "Intermediate (Regular use for specific tasks)",
                    4: "Advanced (Integrate AI frequently in workflow)",
                    5: "Expert (Advanced usage and prompting)"
                }
                
                ai_exp_val = st.slider(
                    "Experience with AI in research",
                    min_value=1, max_value=5, value=st.session_state.get("ai_exp_val", 3),
                    key="ai_exp_slider"
                )
                st.markdown(f"<p style='color: #ffd700; font-size: 0.9em; margin-top: -10px;'>Experience level: <b>{ai_exp_levels[ai_exp_val]}</b></p>", unsafe_allow_html=True)

            p_info = st.text_area(
                "Additional Info (optional)",
                value=st.session_state.get("participant_info", ""),
                placeholder="e.g. Affiliation, field of study…",
                height=80,
            )
            p_id = st.text_input(
                "Participant ID",
                value=st.session_state.proposed_id,
                help="Auto-generated. Change it if you already have one.",
            )

            st.markdown("---")
            st.markdown("### Study Configuration")

            task_id = st.selectbox(
                "Task",
                ["T1 (Targeted Literature Search)", "T2 (Deep Understanding of one paper)"],
                help="Select the task assigned to you.",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start Session", type="primary", width='stretch'):
            if not p_id.strip():
                st.error("Participant ID cannot be empty.")
            else:
                st.session_state.participant_id = p_id.strip()
                st.session_state.participant_name = p_name.strip()
                st.session_state.participant_info = p_info.strip()
                st.session_state.lit_exp_val = lit_exp_val
                st.session_state.ai_exp_val = ai_exp_val
                st.session_state.task_id = task_id
                st.session_state.session_start_ts = datetime.now(timezone.utc).isoformat()
                log_event("task_start", {
                    "task_name": task_id,
                    "lit_review_experience": lit_exp_val,
                    "ai_experience": ai_exp_val
                })
                st.session_state.page = "home"
                st.rerun()


def render_homepage():
    """Render the welcome homepage with mode selection."""
    p_name = st.session_state.get("participant_name", "")
    greeting = f", {p_name}" if p_name else ""
    st.markdown(f"""
        <div style="text-align: center; padding: 4rem 2rem;">
            <h1 class="main-header">Literature Review Dashboard</h1>
            <p class="sub-header" style="font-size: 1.3rem; max-width: 700px; margin: 0 auto 3rem auto;">
                Welcome{greeting}! Explore, analyze, and discover insights from your research papers.
                Choose your preferred mode to get started.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Choose Your Experience")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="metric-card" style="height: 100%;">
                <h3 style="color: #f0f0f0; margin-bottom: 1rem;">Standard Mode</h3>
                <p style="color: #b0b0b0; margin-bottom: 1.5rem;">
                    Browse and analyze your literature collection with powerful filters, 
                    visualizations, and keyword analysis.
                </p>
                <ul style="color: #888888; padding-left: 1.2rem;">
                    <li>Paper browsing & filtering</li>
                    <li>Publication analytics</li>
                    <li>Keyword word cloud & analysis</li>
                    <li>Literature summary & export</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Standard Mode", type="secondary", width='stretch'):
            st.session_state.ai_mode = False
            st.session_state.page = "task_briefing"
            st.rerun()
    
    with col2:
        st.markdown("""
            <div class="metric-card" style="height: 100%; border-color: rgba(255,215,0,0.3);">
                <h3 style="color: #ffd700; margin-bottom: 1rem;">AI Mode</h3>
                <p style="color: #b0b0b0; margin-bottom: 1.5rem;">
                    Unlock AI-powered features with Gemini to get intelligent insights, 
                    summaries, and research assistance.
                </p>
                <ul style="color: #888888; padding-left: 1.2rem;">
                    <li>Everything in Standard Mode</li>
                    <li style="color: #ffd700;">AI-powered paper Q&A</li>
                    <li style="color: #ffd700;">Automated literature summaries</li>
                    <li style="color: #ffd700;">Research insights & gap analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Enter AI Mode", type="primary", width='stretch'):
            st.session_state.used_ai_mode = True
            st.session_state.ai_mode = True
            st.session_state.page = "task_briefing"
            st.rerun()

def render_api_key_input():
    """Render the API key input page."""
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 class="main-header">API Keys Configuration</h1>
            <p class="sub-header">Enter your API keys to enable AI and search features</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #f0f0f0; margin-bottom: 1rem;">Gemini API Key (Required for AI Mode)</h4>
                <p style="color: #888888; margin-bottom: 1rem;">
                    Get a free Gemini API key from Google AI Studio:
                </p>
                <p style="margin-bottom: 1rem;">
                    <a href="https://aistudio.google.com/apikey" target="_blank" style="color: #ffd700;">
                        https://aistudio.google.com/apikey
                    </a>
                </p>
            </div>
        """, unsafe_allow_html=True)

        gemini_key = st.text_input(
            "Gemini API Key",
            type="password",
            placeholder="Enter your Gemini API key here...",
            help="Required for AI-powered features",
            key="gemini_key_input"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #f0f0f0; margin-bottom: 1rem;">Semantic Scholar API Key (Optional but Recommended)</h4>
                <p style="color: #888888; margin-bottom: 1rem;">
                    Get a free Semantic Scholar API key for higher rate limits:
                </p>
                <p style="margin-bottom: 1rem;">
                    <a href="https://www.semanticscholar.org/product/api#api-key" target="_blank" style="color: #ffd700;">
                        https://www.semanticscholar.org/product/api
                    </a>
                </p>
                <p style="color: #888888; font-size: 0.9rem;">
                    Without an API key, you'll be limited to 100 requests per 5 minutes. 
                    With a key, you get 5,000 requests per 5 minutes.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        ss_key = st.text_input(
            "Semantic Scholar API Key (Optional)",
            type="password",
            placeholder="Enter your Semantic Scholar API key here...",
            help="Optional: Improves search rate limits",
            key="ss_key_input"
        )

        st.markdown("""
            <p style="color: #ff9800; font-size: 0.9rem; margin-top: 1rem; text-align: center;">
                WARNING: Your API keys will be stored in memory for this session only and will not be saved to disk.
            </p>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("Back to Home", width='stretch'):
                st.session_state.page = "home"
                st.rerun()
        
        with col_b:
            if st.button("Skip for Now", width='stretch'):
                st.session_state.page = "dashboard"
                st.session_state.ai_mode = True
                st.rerun()

        with col_c:
            if st.button("Save & Continue", type="primary", width='stretch'):
                if not gemini_key:
                    st.error("Please enter at least the Gemini API key to use AI features.")
                else:
                    with st.spinner("Validating API keys..."):
                        gemini_valid = validate_api_key(gemini_key)

                        if gemini_valid:
                            save_api_key_to_session(gemini_key)

                            if ss_key:
                                save_semantic_scholar_key_to_session(ss_key)
                                st.success("Both API keys stored in memory!")
                            else:
                                st.success("Gemini API key stored in memory!")
                                st.info("Semantic Scholar searches will use public rate limits (100 requests/5min)")

                            st.session_state.page = "dashboard"
                            st.session_state.ai_mode = True
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid Gemini API key. Please check and try again.")

def render_ss_api_key_input():
    """Render the Semantic Scholar API key input page for Standard Mode."""
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 class="main-header">Semantic Scholar API Key</h1>
            <p class="sub-header">Enter your API key for better search experience (optional)</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
            <div class="metric-card">
                <h4 style="color: #f0f0f0; margin-bottom: 1rem;">Semantic Scholar API Key (Optional but Recommended)</h4>
                <p style="color: #888888; margin-bottom: 1rem;">
                    Get a free Semantic Scholar API key for higher rate limits:
                </p>
                <p style="margin-bottom: 1rem;">
                    <a href="https://www.semanticscholar.org/product/api#api-key" target="_blank" style="color: #ffd700;">
                        https://www.semanticscholar.org/product/api
                    </a>
                </p>
                <p style="color: #888888; font-size: 0.9rem;">
                    Without an API key, you'll be limited to 100 requests per 5 minutes. 
                    With a key, you get 5,000 requests per 5 minutes.
                </p>
            </div>
        """, unsafe_allow_html=True)

        ss_key = st.text_input(
            "Semantic Scholar API Key",
            type="password",
            placeholder="Enter your Semantic Scholar API key here...",
            help="Optional: Improves search rate limits",
            key="ss_key_input_standard"
        )

        st.markdown("""
            <p style="color: #ff9800; font-size: 0.9rem; margin-top: 1rem; text-align: center;">
                NOTE: Your API key will be stored in memory for this session only and will not be saved to disk.
            </p>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.button("Back to Home", width='stretch'):
                st.session_state.page = "home"
                st.rerun()

        with col_b:
            if st.button("Skip for Now", width='stretch'):
                st.session_state.page = "dashboard"
                st.session_state.ai_mode = False
                st.rerun()

        with col_c:
            if st.button("Save & Continue", type="primary", width='stretch'):
                if ss_key:
                    save_semantic_scholar_key_to_session(ss_key)
                    st.success("Semantic Scholar API key stored in memory!")
                    time.sleep(1)
                st.session_state.page = "dashboard"
                st.session_state.ai_mode = False
                st.rerun()

def search_semantic_scholar(query: str, limit: int = 10) -> list:
    """Search for papers using Semantic Scholar API."""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,year,venue,abstract,externalIds,url,journal"
        }
        
        headers = {"User-Agent": "LiteratureReviewDashboard/1.0"}
        api_key = get_semantic_scholar_key()
        if api_key:
            headers["x-api-key"] = api_key
            
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            papers = []
            for item in data.get('data', []):
                raw_authors = item.get('authors') or []
                authors = [a.get('name', 'Unknown') for a in raw_authors]
                
                venue = item.get('venue') or ''
                journal_info = item.get('journal') or {}
                journal_name = journal_info.get('name', '')
                display_journal = venue or journal_name or 'Unknown'
                paper = {
                    'id': item.get('paperId'),
                    'title': item.get('title', 'Untitled'),
                    'authors': authors,
                    'year': item.get('year'),
                    'journal': display_journal,
                    'abstract': item.get('abstract') or 'No abstract available.',
                    'keywords': [],
                    'url': item.get('url')
                }
                papers.append(paper)
            return papers
        elif response.status_code == 429:
            if not api_key:
                st.warning("Semantic Scholar rate limit reached (100 requests/5min). Add an API key to get 5,000 requests/5min!")
                if st.button("Add Semantic Scholar API Key", type="primary", key="add_ss_key_rate_limit"):
                    st.session_state.page = "api_key"
                    st.rerun()
            else:
                st.warning("Semantic Scholar API rate limit reached. Please wait a moment before searching again.")
            return []
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return []
    except Exception as e:
        st.error(f"Error searching Semantic Scholar: {e}")
        return []

def extract_search_keywords(description: str) -> list:
    """Use Gemini to extract and rephrase search keywords from a research description."""
    prompt = (
        "You are a research librarian. Given the following research description, "
        "produce 3 to 4 diverse keyword sets that can be used to search academic databases. "
        "The first set should be direct keywords from the description. "
        "The remaining sets should be rephrased or synonym variants for broader coverage.\n\n"
        "Return ONLY a JSON array of arrays. Each inner array is a list of keyword strings. "
        "Example: [[\"attention mechanism\", \"image segmentation\"], [\"self-attention\", \"semantic segmentation\"]]\n\n"
        f"Research description: {description}"
    )
    try:
        response = get_gemini_response(prompt)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
        keyword_sets = json.loads(text)
        if isinstance(keyword_sets, list) and all(isinstance(s, list) for s in keyword_sets):
            return keyword_sets
    except Exception:
        pass
    words = description.split()
    chunk_size = min(4, len(words))
    if chunk_size == 0:
        return [[description]]
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(words[i:i + chunk_size])
    return [chunks[0]] if chunks else [[description]]


def search_and_collect(keyword_sets: list, limit_per_query: int = 10) -> list:
    """Search Semantic Scholar with multiple keyword sets and collect all results."""
    all_papers = []
    for kw_set in keyword_sets:
        query = " ".join(kw_set)
        results = search_semantic_scholar(query, limit=limit_per_query)
        all_papers.extend(results)
        time.sleep(0.5)
    return all_papers


def deduplicate_papers(papers: list) -> list:
    """Remove duplicate papers by paperId or normalized title."""
    seen_ids = set()
    seen_titles = set()
    unique = []
    for paper in papers:
        pid = paper.get('id')
        norm_title = (paper.get('title') or '').strip().lower()
        if pid and pid in seen_ids:
            continue
        if norm_title and norm_title in seen_titles:
            continue
        if pid:
            seen_ids.add(pid)
        if norm_title:
            seen_titles.add(norm_title)
        unique.append(paper)
    return unique


def filter_by_relevance(papers: list, description: str, progress_callback=None) -> list:
    """Use Gemini to judge each paper's relevance against the user's research description."""
    aligned = []
    total = len(papers)
    for i, paper in enumerate(papers):
        title = paper.get('title', 'Untitled')
        abstract = paper.get('abstract', 'No abstract available.')
        prompt = (
            "You are an academic research assistant. Determine whether the following paper "
            "is relevant to the user's research description.\n\n"
            f"User's research description: {description}\n\n"
            f"Paper title: {title}\n"
            f"Paper abstract: {abstract}\n\n"
            "Respond with EXACTLY one line starting with either RELEVANT or NOT_RELEVANT, "
            "followed by a brief reason. Example:\n"
            "RELEVANT: This paper directly addresses attention mechanisms for segmentation."
        )
        try:
            response = get_gemini_response(prompt)
            verdict = response.strip().split('\n')[0]
            is_relevant = verdict.upper().startswith('RELEVANT') and not verdict.upper().startswith('NOT_RELEVANT')
        except Exception:
            is_relevant = True
            verdict = "KEPT (error during evaluation)"

        if is_relevant:
            aligned.append(paper)

        if progress_callback:
            progress_callback(i, total, title, is_relevant, verdict)

    return aligned


def render_dashboard():
    """Render the main dashboard."""
    ai_mode = st.session_state.get('ai_mode', False)
    if ai_mode:
        st.session_state.used_ai_mode = True
    with st.sidebar:
        # ── Participant info ──
        p_name = st.session_state.get('participant_name', '')
        p_id = st.session_state.get('participant_id', 'N/A')
        st.markdown(f"**Participant:** {p_name or 'Anonymous'}")
        st.markdown(f"**ID:** `{p_id}`")
        task_label = st.session_state.get('task_id', '')
        st.caption(f"Task: {task_label}")
        st.markdown("---")

        st.markdown("### Navigation")
        if st.button("← Back to Home", width='stretch'):
            st.session_state.page = "home"
            st.rerun()
        st.markdown("---")
        mode_label = "AI Mode" if ai_mode else "Standard Mode"
        st.markdown(f"**Current Mode:** {mode_label}")
        
        if ai_mode:
            if init_gemini():
                st.success("Gemini Connected!")
            else:
                st.error("Gemini API not configured")
                if st.button("Configure API Keys", type="primary", key="sidebar_api_key"):
                    st.session_state.page = "api_key"
                    st.rerun()
            ss_key = get_semantic_scholar_key()
            if ss_key:
                st.success("Semantic Scholar API configured")
            else:
                st.warning("Semantic Scholar: Using public limits")
                st.caption("Configure API key for higher rate limits")

        st.markdown("---")
        with st.expander("Quick Guide", expanded=False):
            st.markdown("""
                **How to use this dashboard:**
                1. **Search Online:** Find new papers and add them to your collection.
                2. **Papers:** View, filter, and sort your collected papers.
                3. **Analytics/Keywords:** Discover trends and common themes in your collection.
                4. **AI Assistant:** Select papers to generate summaries, ask questions, or run Deep Research.
            """)
        st.markdown("---")

        # ── End Session button ──
        if st.button("End Session", type="primary", width='stretch'):
            # Compute duration from session start
            start_ts = st.session_state.get('session_start_ts')
            duration = None
            if start_ts:
                t0 = datetime.fromisoformat(start_ts)
                duration = (datetime.now(timezone.utc) - t0).total_seconds()
            log_event("task_submit", {
                "task_name": st.session_state.get('task_id', ''),
                "duration_seconds": duration,
            })
            st.session_state.page = "session_end"
            st.rerun()

        st.markdown("---")
        st.markdown("#### About")
        st.markdown("A literature review dashboard for exploring research papers.")
    
    st.markdown('<h1 class="main-header">Literature Review Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore, analyze, and discover insights from your research papers</p>', unsafe_allow_html=True)
    
    data_path = Path(__file__).parent / "data" / "example-bib.json"
    
    if not data_path.exists():
        st.error("Data file not found. Please ensure 'data/example-bib.json' exists.")
        return
    
    papers = st.session_state.user_papers
    df = create_dataframe(papers)

    with st.expander("**Filters & Search**", expanded=False):
        years = sorted(df['year'].dropna().unique())
        st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <span class="quick-stat">Total: {len(papers)} papers</span>
                <span class="quick-stat">Years: {min(years)} - {max(years)}</span>
                <span class="quick-stat">Journals: {df['journal'].nunique()}</span>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(years) > 1:
                year_range = st.slider(
                    "Publication Year",
                    min_value=int(min(years)),
                    max_value=int(max(years)),
                    value=(int(min(years)), int(max(years)))
                )
            else:
                year_range = (years[0], years[0]) if years else (2000, 2024)
        
        with col2:
            journals = ['All'] + sorted(df['journal'].unique().tolist())
            selected_journal = st.selectbox("Journal", journals)
        
        with col3:
            search_query = st.text_input("Search papers", placeholder="Search in titles and abstracts...")
        
        all_keywords = set()
        for kws in df['keywords']:
            all_keywords.update(kws)
        selected_keywords = st.multiselect("Filter by Keywords", sorted(all_keywords))
    
    filtered_df = df.copy()
    filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]
    
    if selected_journal != 'All':
        filtered_df = filtered_df[filtered_df['journal'] == selected_journal]
    
    if selected_keywords:
        mask = filtered_df['keywords'].apply(lambda x: any(kw in x for kw in selected_keywords))
        filtered_df = filtered_df[mask]
    
    if search_query:
        mask = (
            filtered_df['title'].str.lower().str.contains(search_query.lower()) |
            filtered_df['abstract'].str.lower().str.contains(search_query.lower())
        )
        filtered_df = filtered_df[mask]
    
    filtered_paper_ids = set(filtered_df['id'].tolist())
    filtered_papers = [p for p in papers if p.get('id') in filtered_paper_ids]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(len(filtered_papers), "Papers Found", "article")
    with col2:
        unique_authors = set()
        for p in filtered_papers:
            unique_authors.update(p.get('authors', []))
        render_metric_card(len(unique_authors), "Unique Authors", "group")
    with col3:
        render_metric_card(filtered_df['journal'].nunique(), "Journals", "menu_book")
    with col4:
        all_kws = set()
        for p in filtered_papers:
            all_kws.update(p.get('keywords', []))
        render_metric_card(len(all_kws), "Keywords", "tag")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    ai_mode = st.session_state.get('ai_mode', False)
    
    tabs_list = ["Papers", "Analytics", "Keywords", "Summary", "Online Search"]
    if ai_mode:
        tabs_list.append("AI Assistant")
    
    tabs = st.tabs(tabs_list)
    
    with tabs[0]:
        st.markdown("### Paper Collection")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                ["Year (Newest First)", "Year (Oldest First)", "Title (A-Z)", "Title (Z-A)", "Author Count"],
                help="Choose how you want to order your paper collection."
            )
        
        sorted_papers = filtered_papers.copy()
        if sort_by == "Year (Newest First)":
            sorted_papers.sort(key=lambda x: x.get('year', 0), reverse=True)
        elif sort_by == "Year (Oldest First)":
            sorted_papers.sort(key=lambda x: x.get('year', 0))
        elif sort_by == "Title (A-Z)":
            sorted_papers.sort(key=lambda x: x.get('title', '').lower())
        elif sort_by == "Title (Z-A)":
            sorted_papers.sort(key=lambda x: x.get('title', '').lower(), reverse=True)
        elif sort_by == "Author Count":
            sorted_papers.sort(key=lambda x: len(x.get('authors', [])), reverse=True)
        if sorted_papers:
            for paper in sorted_papers:
                render_paper_card(paper)
                if st.button("Remove from Collection", key=f"remove_{paper.get('id')}", type="secondary"):
                    remove_paper_from_collection(paper.get('id'))
                    st.rerun()
        else:
            st.info("No papers match your filter criteria.")
    
    with tabs[1]:
        st.markdown("### Publication Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Publications by Year")
            year_counts = filtered_df.groupby('year').size().reset_index(name='count')
            st.bar_chart(year_counts.set_index('year')['count'], color='#ffd700')
        
        with col2:
            st.markdown("#### Papers per Journal")
            journal_counts = filtered_df['journal'].value_counts().head(10)
            st.bar_chart(journal_counts, color='#888888')
        
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Author Collaboration Size")
            author_count_dist = filtered_df['author_count'].value_counts().sort_index()
            st.bar_chart(author_count_dist, color='#ffd700')
        
        with col4:
            st.markdown("#### Keywords per Paper")
            kw_count_dist = filtered_df['keyword_count'].value_counts().sort_index()
            st.bar_chart(kw_count_dist, color='#888888')
    
    with tabs[2]:
        st.markdown("### Keyword Analysis")
        keyword_counter = Counter()
        for p in filtered_papers:
            keyword_counter.update(p.get('keywords', []))
        
        if keyword_counter:
            st.markdown("#### Keyword Word Cloud")
            
            def yellow_grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                import random
                colors = ['#ffd700', '#ffb800', '#e6c200', '#ccac00', '#b39700', '#999999', '#888888', '#777777']
                return random.choice(colors)
            
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='#1a1a1a',
                color_func=yellow_grey_color_func,
                max_words=50,
                prefer_horizontal=0.7,
                min_font_size=10,
                max_font_size=80
            ).generate_from_frequencies(dict(keyword_counter))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            fig.patch.set_facecolor('#1a1a1a')
            st.pyplot(fig)
            plt.close()
            
            st.markdown("---")
            
            st.markdown("#### Most Common Keywords")
            top_keywords = keyword_counter.most_common(15)
            kw_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])
            kw_df = kw_df.set_index('Keyword')
            st.bar_chart(kw_df['Count'], color='#ffd700')
            
            st.markdown("---")
            
            st.markdown("#### Keyword Co-occurrence Matrix")
            st.caption("Papers that share common keywords")
            
            top_10_kws = [kw for kw, _ in keyword_counter.most_common(10)]
            cooccurrence = pd.DataFrame(0, index=top_10_kws, columns=top_10_kws)
            
            for p in filtered_papers:
                paper_kws = set(p.get('keywords', [])) & set(top_10_kws)
                for kw1 in paper_kws:
                    for kw2 in paper_kws:
                        cooccurrence.loc[kw1, kw2] += 1
            
            st.dataframe(cooccurrence, width='stretch')
        else:
            st.info("No keywords available for the filtered papers.")
    
    with tabs[3]:
        st.markdown("### Literature Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Dataset Overview")
            summary_data = {
                "Metric": [
                    "Total Papers",
                    "Publication Period",
                    "Unique Journals",
                    "Unique Authors",
                    "Total Keywords",
                    "Avg. Authors per Paper",
                    "Avg. Keywords per Paper"
                ],
                "Value": [
                    str(len(filtered_papers)),
                    f"{filtered_df['year'].min()} - {filtered_df['year'].max()}",
                    str(filtered_df['journal'].nunique()),
                    str(len(set(a for p in filtered_papers for a in p.get('authors', [])))),
                    str(len(set(k for p in filtered_papers for k in p.get('keywords', [])))),
                    f"{filtered_df['author_count'].mean():.1f}",
                    f"{filtered_df['keyword_count'].mean():.1f}"
                ]
            }
            st.table(pd.DataFrame(summary_data))
        
        with col2:
            st.markdown("#### Top Contributing Authors")
            author_counter = Counter()
            for p in filtered_papers:
                author_counter.update(p.get('authors', []))
            
            top_authors = author_counter.most_common(10)
            if top_authors:
                author_df = pd.DataFrame(top_authors, columns=['Author', 'Papers'])
                st.table(author_df)
        
        st.markdown("---")
        
        st.markdown("#### Full Paper List")
        export_df = filtered_df[['title', 'authors', 'year', 'journal', 'doi']].copy()
        export_df.columns = ['Title', 'Authors', 'Year', 'Journal', 'DOI']
        st.dataframe(export_df, width='stretch')
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="literature_review.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("### Submit Your Task Outputs")
        st.caption("Use the forms below to submit your final findings for this session.")

        with st.expander("Submit Summary", expanded=False):
            summary_text = st.text_area(
                "Write your literature summary",
                height=200,
                key="task_summary_text",
                placeholder="Write a concise summary of the literature you reviewed…",
            )
            if st.button("Submit Summary", type="primary", key="btn_submit_summary"):
                if summary_text.strip():
                    log_event("summary_submit", {
                        "word_count": len(summary_text.split()),
                        "text": summary_text.strip(),
                    })
                    st.success("Summary submitted and logged!")
                else:
                    st.warning("Please write something before submitting.")

        with st.expander("Submit Research Gaps", expanded=False):
            gap_text = st.text_area(
                "Describe research gaps you identified",
                height=200,
                key="task_gap_text",
                placeholder="What gaps or future directions did you identify?…",
            )
            if st.button("Submit Research Gaps", type="primary", key="btn_submit_gap"):
                if gap_text.strip():
                    log_event("gap_submit", {
                        "word_count": len(gap_text.split()),
                        "text": gap_text.strip(),
                    })
                    st.success("Research gaps submitted and logged!")
                else:
                    st.warning("Please write something before submitting.")

        with st.expander("Submit Keywords (Task T2)", expanded=False):
            kw_input = st.text_input(
                "Enter keywords separated by commas",
                key="task_keywords_input",
                placeholder="e.g. attention mechanism, segmentation, U-Net",
            )
            if st.button("Submit Keywords", type="primary", key="btn_submit_kw"):
                kw_list = [k.strip() for k in kw_input.split(",") if k.strip()]
                if kw_list:
                    log_event("keywords_submit", {
                        "keywords": kw_list,
                    })
                    st.success(f"Submitted {len(kw_list)} keywords!")
                else:
                    st.warning("Please enter at least one keyword.")
    
    with tabs[4]:
        st.markdown("### Search Semantic Scholar")
        st.markdown("Find real academic papers online and add them to your knowledge base.")
        
        col_s1, col_s2 = st.columns([3, 1])
        with col_s1:
            search_term = st.text_input("Enter search query", placeholder="e.g. deep learning for medical imaging")
        with col_s2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_button = st.button("Search Online", type="primary", width='stretch')
            
        if search_button and search_term:
            prev = st.session_state.get('last_search_query')
            if prev and prev != search_term:
                log_event("keyword_refine", {
                    "previous_query": prev,
                    "new_query": search_term,
                })
            log_event("search_query", {
                "query_text": search_term,
                "query_length": len(search_term),
            })
            st.session_state.last_search_query = search_term

            with st.spinner("Searching Semantic Scholar..."):
                results = search_semantic_scholar(search_term)
                st.session_state.search_results = results
                st.session_state.search_term_used = search_term
        
        search_results = st.session_state.get('search_results', [])
        if search_results:
            st.success(f"Found {len(search_results)} papers for '{st.session_state.get('search_term_used', '')}'.")
            existing_titles = {p.get('title', '').lower() for p in st.session_state.get('user_papers', [])}
            for idx, paper in enumerate(search_results):
                with st.expander(f"{paper.get('title', 'Untitled')}", expanded=False):
                    log_event("paper_open", {
                        "paper_id": str(paper.get('id', '')),
                        "rank_position": idx + 1,
                    })
                    render_paper_card(paper)
                    col_link, col_add = st.columns([3, 1])
                    with col_link:
                        if paper.get('url'):
                            if st.link_button("View on Semantic Scholar", paper.get('url')):
                                pass
                            log_event("source_verification_click", {
                                "paper_id": str(paper.get('id', '')),
                                "source_type": "external_link",
                            })
                    with col_add:
                        already_added = paper.get('title', '').lower() in existing_titles
                        if already_added:
                            st.button("Already in Collection", key=f"add_{paper.get('id')}", disabled=True)
                        else:
                            if st.button("Add to Knowledge Base", key=f"add_{paper.get('id')}", type="primary"):
                                log_event("paper_select", {
                                    "paper_id": str(paper.get('id', '')),
                                })
                                add_paper_to_collection(paper)
                                st.toast(f"Added: {paper.get('title')}")
                                st.rerun()
        elif search_button:
            st.warning("No results found or error occurred.")

    if ai_mode:
        with tabs[5]:
            st.markdown("### AI Research Assistant")
            
            if not init_gemini():
                st.warning("API Key Required")
                st.info("To use AI features, please provide your Gemini API key. The key will be stored in memory for this session only and will not be saved to disk.")

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Enter API Key", type="primary", key="ai_tab_api_key", width='stretch'):
                        st.session_state.page = "api_key"
                        st.rerun()
            else:
                col_nav, col_content = st.columns([1, 4])
                
                with col_nav:
                  with st.container(border=True):
                    st.markdown('<div class="ai-nav-title">AI Tools</div>', unsafe_allow_html=True)
                    selected_tool = st.radio(
                        "Tool Selection",
                        ["Paper Chat", "AI Summary", "Research Insights", "Deep Research"],
                        label_visibility="collapsed"
                    )
                    tool_descriptions = {
                        "Paper Chat": "Ask questions about a specific paper.",
                        "AI Summary": "Generate summaries across multiple papers.",
                        "Research Insights": "Discover themes and citation patterns.",
                        "Deep Research": "AI-powered paper discovery from the web."
                    }
                    st.markdown(f'<div class="ai-tool-desc">{tool_descriptions[selected_tool]}</div>', unsafe_allow_html=True)
                
                with col_content:
                  with st.container(border=True):
                    if selected_tool == "Paper Chat":
                        st.markdown('<div class="ai-content-header">Ask Questions About a Paper</div>', unsafe_allow_html=True)
                        st.markdown('<div class="ai-content-subtitle">Select a paper and ask the AI anything about it.</div>', unsafe_allow_html=True)
                        
                        paper_titles = [p.get('title', 'Untitled') for p in filtered_papers]
                        if paper_titles:
                            selected_title = st.selectbox("Select a paper", paper_titles, key="chat_paper_select")
                            selected_paper = next((p for p in filtered_papers if p.get('title') == selected_title), None)
                            
                            if selected_paper:
                                with st.expander("Selected Paper Details", expanded=False):
                                    render_paper_card(selected_paper)
                                
                                chat_context = f"Title: {selected_paper.get('title')}\nAuthors: {', '.join(selected_paper.get('authors', []))}\nAbstract: {selected_paper.get('abstract', 'N/A')}\nKeywords: {', '.join(selected_paper.get('keywords', []))}"
                            
                                user_question = st.text_area(
                                    "Your Question",
                                    placeholder="e.g., What methodology does this paper use? How does it evaluate the results?",
                                    height=100,
                                    help="Type your question here. The AI will use the paper's title, abstract, and authors to formulate an answer."
                                )
                                
                                if st.button("Ask AI", type="primary", help="Click to get an AI-generated answer based on the context of the selected paper."):
                                    if user_question:
                                        log_event("ai_call", {
                                            "feature": "qa",
                                            "input_length": len(user_question),
                                        })
                                        with st.spinner("Analyzing paper..."):
                                            context = f"You are a research assistant. Analyze the following academic paper and answer the user's question.\n\nPaper:\n{chat_context}"
                                            response = get_gemini_response(user_question, context)
                                            log_event("ai_output_generated", {
                                                "feature": "qa",
                                                "output_length": len(response),
                                            })
                                            st.markdown("#### Response")
                                            st.markdown(response)
                                    else:
                                        st.warning("Please enter a question.")
                        else:
                            st.info("No papers available. Adjust your filters to include papers.")
                    
                    elif selected_tool == "AI Summary":
                        st.markdown('<div class="ai-content-header">Generate AI Summaries</div>', unsafe_allow_html=True)
                        st.markdown('<div class="ai-content-subtitle">Select papers and choose a summary type to generate insights.</div>', unsafe_allow_html=True)
                        
                        summary_paper_titles = [p.get('title', 'Untitled') for p in filtered_papers]
                        selected_summary_papers = st.multiselect(
                            "Select papers to include in summary",
                            summary_paper_titles,
                            default=summary_paper_titles[:10],
                            key="summary_paper_select"
                        )
                        st.caption(f"Analyzing {len(selected_summary_papers)} of {len(filtered_papers)} papers.")
                        
                        summary_type = st.selectbox(
                            "Summary Type",
                            ["Literature Overview", "Research Gaps", "Methodology Comparison", "Key Findings"],
                            help="Select the type of AI-generated summary you want for the selected papers."
                        )
                        
                        if st.button("Generate Summary", type="primary", help="Click to generate an AI summary based on the abstracts of the selected papers."):
                            if not selected_summary_papers:
                                st.warning("Please select at least one paper.")
                            else:
                                log_event("ai_call", {
                                    "feature": "summary",
                                    "input_length": len(selected_summary_papers),
                                })
                                with st.spinner(f"Generating {summary_type}..."):
                                    selected_papers_data = [p for p in filtered_papers if p.get('title') in selected_summary_papers]
                                    summary_context = "\n\n".join([
                                        f"Title: {p.get('title')}\nAuthors: {', '.join(p.get('authors', []))}\nAbstract: {p.get('abstract', 'N/A')}\nKeywords: {', '.join(p.get('keywords', []))}"
                                        for p in selected_papers_data
                                    ])
                                    prompts = {
                                        "Literature Overview": "Provide a comprehensive literature overview summarizing the main themes, research areas, and contributions of these papers.",
                                        "Research Gaps": "Identify potential research gaps and future research directions based on these papers.",
                                        "Methodology Comparison": "Compare and contrast the research methodologies used across these papers.",
                                        "Key Findings": "Summarize the key findings and conclusions from each paper."
                                    }
                                    context = f"You are an academic research analyst. Based on the following papers, {prompts[summary_type].lower()}\n\nPapers:\n{summary_context}"
                                    response = get_gemini_response(prompts[summary_type], context)
                                    log_event("ai_output_generated", {
                                        "feature": "summary",
                                        "output_length": len(response),
                                    })
                                    st.markdown(f"#### {summary_type}")
                                    st.markdown(response)
                    
                    elif selected_tool == "Research Insights":
                        st.markdown('<div class="ai-content-header">Research Insights</div>', unsafe_allow_html=True)
                        st.markdown('<div class="ai-content-subtitle">Get AI-generated thematic analysis and citation suggestions for your papers.</div>', unsafe_allow_html=True)
                        
                        insights_paper_titles = [p.get('title', 'Untitled') for p in filtered_papers]
                        selected_insights_papers = st.multiselect(
                            "Select papers to analyze",
                            insights_paper_titles,
                            default=insights_paper_titles[:10],
                            key="insights_paper_select"
                        )
                        st.caption(f"Analyzing {len(selected_insights_papers)} of {len(filtered_papers)} papers.")
                        
                        insights_papers_data = [p for p in filtered_papers if p.get('title') in selected_insights_papers]
                        insights_context = "\n\n".join([
                            f"Title: {p.get('title')}\nAuthors: {', '.join(p.get('authors', []))}\nAbstract: {p.get('abstract', 'N/A')}\nKeywords: {', '.join(p.get('keywords', []))}"
                            for p in insights_papers_data
                        ])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("Thematic Analysis", width='stretch'):
                                if not selected_insights_papers:
                                    st.warning("Please select at least one paper.")
                                else:
                                    log_event("ai_call", {
                                        "feature": "gap_analysis",
                                        "input_length": len(selected_insights_papers),
                                    })
                                    with st.spinner("Analyzing themes..."):
                                        context = f"You are a research analyst. Analyze the following papers and identify major themes and connections.\n\nPapers:\n{insights_context}"
                                        response = get_gemini_response("Identify the major themes and connections between these papers. Create a thematic map of the research.", context)
                                        log_event("ai_output_generated", {
                                            "feature": "gap_analysis",
                                            "output_length": len(response),
                                        })
                                        st.session_state['thematic_response'] = response
                        
                        with col2:
                            if st.button("Citation Suggestions", width='stretch'):
                                if not selected_insights_papers:
                                    st.warning("Please select at least one paper.")
                                else:
                                    log_event("ai_call", {
                                        "feature": "gap_analysis",
                                        "input_length": len(selected_insights_papers),
                                    })
                                    with st.spinner("Generating suggestions..."):
                                        context = f"You are a research advisor. Based on the following papers, suggest how they could be cited together in a literature review.\n\nPapers:\n{insights_context}"
                                        response = get_gemini_response("Suggest how these papers could be organized and cited together in a literature review section.", context)
                                        log_event("ai_output_generated", {
                                            "feature": "gap_analysis",
                                            "output_length": len(response),
                                        })
                                        st.session_state['citation_response'] = response
                        
                        if st.session_state.get('thematic_response'):
                            st.markdown("#### Thematic Analysis")
                            st.markdown(st.session_state['thematic_response'])
                        
                        if st.session_state.get('citation_response'):
                            st.markdown("#### Citation Suggestions")
                            st.markdown(st.session_state['citation_response'])
    
                    elif selected_tool == "Deep Research":
                        st.markdown('<div class="ai-content-header">Deep Research</div>', unsafe_allow_html=True)
                        st.markdown('<div class="ai-content-subtitle">Describe what you\'re looking for in natural language. The AI will extract keywords, search Semantic Scholar, remove duplicates, and filter results by relevance.</div>', unsafe_allow_html=True)
    
                        research_desc = st.text_area(
                            "Research Description",
                            placeholder="e.g., I'm looking for papers about using attention mechanisms for improving medical image segmentation, specifically focusing on U-Net architectures.",
                            height=120,
                            key="deep_research_desc",
                            help="Describe your research topic in detail. The more specific you are, the better the AI can find relevant papers."
                        )
    
                        if st.button("Start Deep Research", type="primary", key="deep_research_btn", help="Click to extract keywords, search online, and filter papers based on your description."):
                            if not research_desc.strip():
                                st.warning("Please enter a research description.")
                            else:
                                log_event("ai_call", {
                                    "feature": "deep_research",
                                    "input_length": len(research_desc),
                                })
                                with st.status("Deep Research in progress...", expanded=True) as status:
                                    st.write("**Step 1/4** - Extracting search keywords from your description...")
                                    keyword_sets = extract_search_keywords(research_desc)
                                    for idx, kw_set in enumerate(keyword_sets):
                                        st.write(f"  Keyword set {idx + 1}: `{', '.join(kw_set)}`")
                                    st.write(f"**Step 2/4** - Searching Semantic Scholar with {len(keyword_sets)} keyword sets...")
                                    raw_papers = search_and_collect(keyword_sets)
                                    st.write(f"  Found **{len(raw_papers)}** raw results.")
    
                                    if not raw_papers:
                                        status.update(label="Deep Research completed - no results found.", state="complete")
                                        st.warning("No papers were found. Try a different description.")
                                    else:
                                        st.write("**Step 3/4** - Removing duplicate papers...")
                                        unique_papers = deduplicate_papers(raw_papers)
                                        removed = len(raw_papers) - len(unique_papers)
                                        st.write(f"  Removed **{removed}** duplicates. **{len(unique_papers)}** unique papers remain.")
    
                                        st.write(f"**Step 4/4** - Evaluating relevance of {len(unique_papers)} papers...")
                                        progress_bar = st.progress(0, text="Evaluating papers...")
                                        verdict_container = st.container()
    
                                        def progress_cb(i, total, title, is_relevant, verdict):
                                            pct = (i + 1) / total
                                            progress_bar.progress(pct, text=f"Evaluated {i + 1}/{total} papers")
                                            icon = "[+]" if is_relevant else "[-]"
                                            short_title = title[:80] + "..." if len(title) > 80 else title
                                            verdict_container.write(f"  {icon} {short_title}")
    
                                        aligned_papers = filter_by_relevance(unique_papers, research_desc, progress_callback=progress_cb)
    
                                        status.update(
                                            label=f"Deep Research complete - {len(aligned_papers)} relevant papers found!",
                                            state="complete"
                                        )
    
                                        st.session_state['deep_research_results'] = aligned_papers
                                        log_event("ai_output_generated", {
                                            "feature": "deep_research",
                                            "output_length": len(aligned_papers),
                                        })
    
                        if st.session_state.get('deep_research_results'):
                            results = st.session_state['deep_research_results']
                            st.markdown(f"### Results ({len(results)} relevant papers)")
                            existing_titles = {p.get('title', '').lower() for p in st.session_state.get('user_papers', [])}
                            for paper in results:
                                with st.expander(f"{paper.get('title', 'Untitled')}", expanded=False):
                                    log_event("deep_research_link_click", {
                                        "paper_id": str(paper.get('id', '')),
                                    })
                                    render_paper_card(paper)
                                    col_link, col_add = st.columns([3, 1])
                                    with col_link:
                                        if paper.get('url'):
                                            if st.link_button("View on Semantic Scholar", paper.get('url')):
                                                pass
                                            log_event("source_verification_click", {
                                                "paper_id": str(paper.get('id', '')),
                                                "source_type": "deep_research_external_link",
                                            })
                                    with col_add:
                                        already_added = paper.get('title', '').lower() in existing_titles
                                        if already_added:
                                            st.button("Already in Collection", key=f"dr_add_{paper.get('id')}", disabled=True)
                                        else:
                                            if st.button("Add to Knowledge Base", key=f"dr_add_{paper.get('id')}", type="primary"):
                                                log_event("paper_select", {
                                                    "paper_id": str(paper.get('id', '')),
                                                    "source": "deep_research"
                                                })
                                                add_paper_to_collection(paper)
                                                st.toast(f"Added: {paper.get('title')}")
                                                st.rerun()


def render_session_end():
    """Post-session surveys and thank-you."""
    p_name = st.session_state.get('participant_name', 'Participant')
    st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <h1 class="main-header">Thank You, {p_name}!</h1>
            <p class="sub-header">Please complete the following surveys before viewing your report.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("### 1. System Usability Scale (SUS)")
        st.caption("Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree).")
        st.info(
            "This section measures how easy and pleasant you found the system to use overall. "
            "There are no right or wrong answers — we are interested in your honest, personal impression. "
            "Some statements are phrased positively and some negatively; please read each one carefully."
        )
        sus_items = [
            ("I think that I would like to use this system frequently.",
             "Consider whether you would choose to use this tool regularly for literature reviews."),
            ("I found the system unnecessarily complex.",
             "Think about whether the interface had too many features or confusing layouts."),
            ("I thought the system was easy to use.",
             "Reflect on how intuitive the controls, navigation, and workflows felt."),
            ("I think that I would need the support of a technical person to use this system.",
             "Consider whether you could use this tool independently without assistance."),
            ("I found the various functions in this system were well integrated.",
             "Think about whether features like search, filtering, and AI tools worked together smoothly."),
            ("I thought there was too much inconsistency in this system.",
             "Consider whether different parts of the interface behaved in unexpected or contradictory ways."),
            ("I would imagine that most people would learn to use this system very quickly.",
             "Think about whether a colleague with no prior training could pick this up easily."),
            ("I found the system very cumbersome to use.",
             "Reflect on whether completing tasks felt awkward, slow, or required too many steps."),
            ("I felt very confident using the system.",
             "Consider how sure you felt about knowing what to do at each step."),
            ("I needed to learn a lot of things before I could get going with this system.",
             "Think about how much upfront learning was required before you could be productive."),
        ]
        sus_responses = {}
        for i, (item, tip) in enumerate(sus_items, 1):
            sus_responses[f"Q{i}"] = st.slider(
                f"Q{i}: {item}", 1, 5, 3, key=f"sus_q{i}",
                help=tip
            )

        st.markdown("---")

        st.markdown("### 2. Cognitive Workload (NASA-TLX)")
        st.caption("Rate each dimension from 1 (Very Low) to 7 (Very High).")
        st.info(
            "This section assesses how demanding the tasks felt across several dimensions. "
            "A score of 1 means the demand was very low, and 7 means it was very high. "
            "For **Performance**, a higher score means you felt more successful."
        )
        tlx_dims = [
            ("Mental Demand", "How mentally demanding was the task?",
             "How much thinking, deciding, calculating, or remembering was required?"),
            ("Physical Demand", "How physically demanding was the task?",
             "How much physical activity was required (e.g. scrolling, clicking, typing)?"),
            ("Temporal Demand", "How hurried or rushed was the pace of the task?",
             "Did you feel time pressure? Was the pace comfortable or stressful?"),
            ("Performance", "How successful were you in accomplishing what you were asked to do?",
             "How satisfied are you with your performance? Higher = more successful."),
            ("Effort", "How hard did you have to work to accomplish your level of performance?",
             "How much mental and physical effort did you have to put in overall?"),
            ("Frustration", "How insecure, discouraged, irritated, stressed were you?",
             "Did you feel annoyed, stressed, or discouraged at any point during the session?"),
        ]
        tlx_responses = {}
        for dim_name, dim_desc, dim_tip in tlx_dims:
            tlx_responses[dim_name] = st.slider(
                f"{dim_name}: {dim_desc}", 1, 7, 4, key=f"tlx_{dim_name}",
                help=dim_tip
            )

        st.markdown("---")

        st.markdown("### 3. Trust in the System")
        st.caption("Rate each statement from 1 (Strongly Disagree) to 7 (Strongly Agree).")
        st.info(
            "This section measures how much you trust the system and its outputs. "
            "Consider factors like reliability, predictability, and whether you would feel comfortable "
            "relying on the system for real research work."
        )
        trust_items = [
            ("The system is reliable.",
             "Did the system work consistently without errors or unexpected behavior?"),
            ("I can trust the information provided by the system.",
             "Did the results, summaries, and suggestions seem accurate and credible?"),
            ("I am confident in the system's outputs.",
             "Would you feel comfortable using the outputs in your actual research?"),
            ("The system behaves in a predictable manner.",
             "Could you anticipate how the system would respond to your actions?"),
            ("I am comfortable relying on the system for my research tasks.",
             "Would you delegate parts of your literature review workflow to this tool?"),
        ]
        trust_responses = {}
        for i, (item, tip) in enumerate(trust_items, 1):
            trust_responses[f"Q{i}"] = st.slider(
                f"Q{i}: {item}", 1, 7, 4, key=f"trust_q{i}",
                help=tip
            )

        st.markdown("---")

        st.markdown("### 4. Fatigue & Engagement")
        st.caption("Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree).")
        st.info(
            "This section captures how tired or engaged you felt during the session. "
            "Your answers help us understand whether the system is sustainable for extended use."
        )
        fatigue_items = [
            ("I feel mentally fatigued after completing the tasks.",
             "Do you feel mentally drained or tired after this session?"),
            ("I remained engaged throughout the session.",
             "Did you stay focused and interested, or did your attention drift?"),
            ("I would be willing to use this tool again for literature review.",
             "Knowing what you know now, would you voluntarily use this tool in the future?"),
        ]
        fatigue_responses = {}
        for i, (item, tip) in enumerate(fatigue_items, 1):
            fatigue_responses[f"Q{i}"] = st.slider(
                f"Q{i}: {item}", 1, 5, 3, key=f"fatigue_q{i}",
                help=tip
            )

        ai_pref_responses = {}
        if st.session_state.get('used_ai_mode', False):
            st.markdown("---")
            st.markdown("### 5. AI Feature Preferences")
            st.caption("Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree).")
            st.info(
                "Since you used AI-assisted features during this session, we'd like to know how helpful they were. "
                "Think about whether the AI tools saved you time, provided trustworthy outputs, and whether "
                "you'd prefer the AI-assisted workflow over doing everything manually."
            )
            ai_pref_items = [
                ("The AI features helped me find relevant papers faster.",
                 "Did AI search, summaries, or deep research save you time compared to manual browsing?"),
                ("I trusted the AI-generated summaries and insights.",
                 "Did you find the AI outputs believable and accurate enough to use?"),
                ("I would prefer using the AI-assisted mode over manual mode.",
                 "Given the choice, would you pick the AI-assisted workflow for future tasks?"),
            ]
            for i, (item, tip) in enumerate(ai_pref_items, 1):
                ai_pref_responses[f"Q{i}"] = st.slider(
                    f"Q{i}: {item}", 1, 5, 3, key=f"ai_pref_q{i}",
                    help=tip
                )

        st.markdown("---")

        if st.button("Submit Surveys & View Report", type="primary", width='stretch'):
            # Log all survey responses
            log_event("survey_response", {
                "instrument": "SUS",
                "responses": sus_responses,
            })
            log_event("survey_response", {
                "instrument": "NASA_TLX",
                "responses": tlx_responses,
            })
            log_event("survey_response", {
                "instrument": "Trust",
                "responses": trust_responses,
            })
            log_event("survey_response", {
                "instrument": "Fatigue",
                "responses": fatigue_responses,
            })
            if ai_pref_responses:
                log_event("survey_response", {
                    "instrument": "AI_Preference",
                    "responses": ai_pref_responses,
                })
            st.session_state.page = "session_report"
            st.rerun()


def render_session_report():
    """Display a research report computed from the participant's event log."""
    p_id = st.session_state.get('participant_id', 'N/A')
    p_name = st.session_state.get('participant_name', 'Participant')

    st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <h1 class="main-header">Session Report</h1>
            <p class="sub-header">Summary for {p_name} (ID: {p_id})</p>
        </div>
    """, unsafe_allow_html=True)

    events = load_participant_log(p_id)
    if not events:
        st.warning("No events found for this participant.")
        return

    metrics = compute_derived_metrics(events)

    st.markdown("### Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        tct = metrics.get('task_completion_time_seconds')
        render_metric_card(
            f"{tct:.0f}s" if tct else "N/A",
            "Task Completion Time", "timer"
        )
    with c2:
        render_metric_card(metrics.get('num_search_queries', 0), "Search Queries", "search")
    with c3:
        render_metric_card(metrics.get('num_papers_opened', 0), "Papers Opened", "open_in_new")
    with c4:
        render_metric_card(metrics.get('num_papers_selected', 0), "Papers Selected", "done_all")

    st.markdown("---")

    st.markdown("### A. Efficiency Metrics")
    eff_data = {
        "Metric": [
            "Task Completion Time (s)",
            "Search Queries",
            "Keyword Refinements",
            "Papers Opened",
            "Papers Selected",
        ],
        "Value": [
            f"{metrics.get('task_completion_time_seconds', 0):.1f}" if metrics.get('task_completion_time_seconds') else "N/A",
            str(metrics.get('num_search_queries', 0)),
            str(metrics.get('num_keyword_refinements', 0)),
            str(metrics.get('num_papers_opened', 0)),
            str(metrics.get('num_papers_selected', 0)),
        ],
    }
    st.table(pd.DataFrame(eff_data))

    st.markdown("---")

    st.markdown("### B. Time Spent per Mode & Section")
    time_metrics = metrics.get("time_metrics", {})
    time_data = []
    for cond, sections in time_metrics.items():
        for sec, t_spent in sections.items():
            time_data.append({"Mode": cond, "Section": sec, "Time (s)": round(t_spent, 1)})
    if time_data:
        time_df = pd.DataFrame(time_data)
        time_pivot = time_df.pivot(index="Section", columns="Mode", values="Time (s)").fillna(0)
        st.dataframe(time_pivot, width='stretch')
        
        st.markdown("**Total Time per Mode:**")
        mode_totals = {cond: sum(sections.values()) for cond, sections in time_metrics.items()}
        cols = st.columns(max(len(mode_totals), 1))
        for i, (cond, total) in enumerate(mode_totals.items()):
            with cols[i]:
                render_metric_card(f"{total:.1f}s", cond, "schedule")
    else:
        st.info("No time tracking data available.")

    st.markdown("---")

    st.markdown("### C. AI Usage")
    ai_col1, ai_col2 = st.columns(2)
    with ai_col1:
        ratio = metrics.get('ai_reliance_ratio')
        render_metric_card(
            f"{ratio:.2%}" if ratio is not None else "N/A",
            "AI Reliance Ratio", "robot"
        )
    with ai_col2:
        breakdown = metrics.get('ai_feature_breakdown', {})
        if breakdown:
            feat_df = pd.DataFrame(
                list(breakdown.items()), columns=["Feature", "Calls"]
            ).set_index("Feature")
            st.bar_chart(feat_df["Calls"], color="#ffd700")
        else:
            st.info("No AI features were used in this session.")

    st.markdown("---")

    st.markdown("### D. Exploration Behavior")
    expl_col1, expl_col2, expl_col3 = st.columns(3)
    with expl_col1:
        depth = metrics.get('exploration_depth')
        render_metric_card(
            f"{depth:.2f}" if depth is not None else "N/A",
            "Exploration Depth (opened / selected)", "explore"
        )
    with expl_col2:
        render_metric_card(
            metrics.get('num_papers_opened', 0),
            "Online Search File Opens", "verified"
        )
    with expl_col3:
        render_metric_card(
            metrics.get('num_deep_research_link_clicks', 0),
            "Deep Research Link Clicks", "ads_click"
        )

    st.markdown("---")

    st.markdown("### E. Trust & Verification")
    vr = metrics.get('verification_rate')
    dr_vr = metrics.get('deep_research_verification_rate')
    
    tr_col1, tr_col2 = st.columns(2)
    with tr_col1:
        render_metric_card(
            f"{vr:.2f}" if vr is not None else "N/A",
            "Verifications per AI Output", "shield"
        )
    with tr_col2:
        render_metric_card(
            f"{dr_vr:.2f}" if dr_vr is not None else "N/A",
            "Verifications per Deep Research Output", "verified_user"
        )

    st.markdown("---")

    st.markdown("### F. Survey Scores")
    survey_col1, survey_col2, survey_col3 = st.columns(3)
    with survey_col1:
        sus = metrics.get('sus_score')
        render_metric_card(
            f"{sus:.1f}" if sus is not None else "N/A",
            "SUS Score (0–100)", "fact_check"
        )
    with survey_col2:
        tlx = metrics.get('nasa_tlx_mean')
        render_metric_card(
            f"{tlx:.2f}" if tlx is not None else "N/A",
            "NASA-TLX Mean (1–7)", "psychology"
        )
    with survey_col3:
        trust = metrics.get('trust_mean')
        render_metric_card(
            f"{trust:.2f}" if trust is not None else "N/A",
            "Trust Mean (1–7)", "handshake"
        )

    st.markdown("---")

    st.markdown("### G. Event Timeline")
    timeline_data = []
    for e in events:
        timeline_data.append({
            "Time": e["timestamp"],
            "Event": e["event_type"],
            "Condition": e["condition"],
        })
    if timeline_data:
        st.dataframe(pd.DataFrame(timeline_data), width='stretch')

    st.markdown("---")

    st.markdown("### H. Export")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_str = events_to_csv_string(events)
        st.download_button(
            "Download Event Log (CSV)",
            data=csv_str,
            file_name=f"{p_id}_events.csv",
            mime="text/csv",
            width='stretch',
        )
    with exp_col2:
        import json as _json
        report_json = _json.dumps({
            "participant_id": p_id,
            "participant_name": p_name,
            "metrics": {k: v for k, v in metrics.items()},
        }, indent=2, default=str)
        st.download_button(
            "Download Summary Report (JSON)",
            data=report_json,
            file_name=f"{p_id}_report.json",
            mime="application/json",
            width='stretch',
        )

    st.markdown("---")
    if st.button("Start New Session", type="primary", width='stretch'):
        # Clear session and restart
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


def render_task_briefing():
    """Render the task sample, criteria, and tool intro."""
    import random
    import sys
    import os
    
    sys.path.append(os.path.dirname(__file__))
    try:
        from task_config import TASKS, TOOL_TUTORIALS
    except ImportError:
        TASKS = {"T1": {"name": "T1", "objective": "", "criteria": [], "samples": [""]}}
        TOOL_TUTORIALS = {"Manual": [], "AI": []}

    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 class="main-header">Task Briefing</h1>
            <p class="sub-header">Review your assigned task and available tools</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Identify task
    task_id_raw = st.session_state.get("task_id", "T1 (Targeted Literature Search)")
    t_id = "T1" if "T1" in task_id_raw else "T2"
    task_info = TASKS.get(t_id, TASKS.get("T1"))
    
    # Randomly select a sample if not already selected
    if 'task_sample' not in st.session_state:
        st.session_state.task_sample = random.choice(task_info["samples"])
        
    sample = st.session_state.task_sample
    
    # Task specific details
    st.markdown(f"### {task_info['name']}")
    st.markdown(f"**Objective:** {task_info['objective']}")
    st.info(f"**Your Topic:**\n\n{sample}")
    
    st.markdown("#### Requirements:")
    for c in task_info['criteria']:
        st.markdown(f"- {c}")
        
    st.markdown("---")
    
    # Tool introduction based on mode
    ai_mode = st.session_state.get("ai_mode", False)
    mode_str = "AI" if ai_mode else "Manual"
    st.markdown(f"### Available Tools ({mode_str} Mode)")
    
    tools = TOOL_TUTORIALS.get(mode_str, TOOL_TUTORIALS.get("Manual", []))
    for t in tools:
        st.markdown(t)
        
    st.markdown("---")
    if st.button("Proceed to Dashboard", type="primary", width='stretch'):
        if ai_mode:
            if init_gemini():
                st.session_state.page = "dashboard"
            else:
                st.session_state.page = "api_key"
        else:
            if not get_semantic_scholar_key():
                st.session_state.page = "ss_api_key"
            else:
                st.session_state.page = "dashboard"
        st.rerun()

def main():
    """Main application entry point with page routing."""
    
    init_session()

    if 'page' not in st.session_state:
        st.session_state.page = "participant_setup"

    # Scroll to top hack to prevent pages loading from the middle
    st.components.v1.html(
        f"""
        <script>
            var body = window.parent.document.querySelector('.main');
            if (body) {{ body.scrollTop = 0; }}
        </script>
        """,
        height=0
    )
    if 'ai_mode' not in st.session_state:
        st.session_state.ai_mode = False
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'search_term_used' not in st.session_state:
        st.session_state.search_term_used = ""
    if 'runtime_api_key' not in st.session_state:
        st.session_state.runtime_api_key = None
    if 'runtime_ss_api_key' not in st.session_state:
        st.session_state.runtime_ss_api_key = None
    if 'user_papers' not in st.session_state:
        data_path = Path(__file__).parent / "data" / "example-bib.json"
        if data_path.exists():
            st.session_state.user_papers = load_data(str(data_path))
        else:
            st.session_state.user_papers = []

    page = st.session_state.page

    if page == "participant_setup":
        render_participant_setup()
    elif page == "home":
        render_homepage()
    elif page == "task_briefing":
        render_task_briefing()
    elif page == "api_key":
        render_api_key_input()
    elif page == "ss_api_key":
        render_ss_api_key_input()
    elif page == "session_end":
        render_session_end()
    elif page == "session_report":
        render_session_report()
    else:
        render_dashboard()

if __name__ == "__main__":
    main()
