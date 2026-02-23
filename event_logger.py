"""
Event Logger for HCI Study — Literature Review Dashboard

Centralized logging module. All events are appended as JSON lines to
`logs/<participant_id>.jsonl`.  Every UI component calls `log_event()`
instead of writing logging logic of its own.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

LOGS_DIR = Path(__file__).parent / "logs"

CONDITION_MAP = {
    False: "A (Manual model)",
    True: "B (AI model)",
}


def generate_participant_id() -> str:
    """Return a short, human-friendly random ID (8-char hex)."""
    return uuid.uuid4().hex[:8]


def init_session():
    """Initialise session-state keys used by the logger.

    Call once at startup (idempotent – safe to call on every rerun).
    """
    defaults = {
        "participant_id": None,
        "participant_name": "",
        "participant_info": "",
        "task_id": "T1 (Targeted Literature Search)",
        "session_start_ts": None,      # ISO-8601 set at task_start
        "last_search_query": None,     # for keyword_refine detection
        "used_ai_mode": False,         # True if AI mode was used at any point
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _current_condition() -> str:
    """Return the condition string based on the *currently active* mode."""
    ai_mode = st.session_state.get("ai_mode", False)
    return CONDITION_MAP.get(ai_mode, "A (Manual model)")


def log_event(event_type: str, event_value=None):
    """Append one event to the participant's JSONL log file.

    Parameters
    ----------
    event_type : str
        One of the canonical event-type strings (e.g. ``"search_query"``).
    event_value : dict | str | int | None
        Arbitrary payload for this event.
    """
    participant_id = st.session_state.get("participant_id")
    if not participant_id:
        return  # session not initialised yet – silently skip

    entry = {
        "participant_id": participant_id,
        "participant_name": st.session_state.get("participant_name", ""),
        "condition": _current_condition(),
        "task_id": st.session_state.get("task_id", ""),
        "section": st.session_state.get("current_section_name", "Home"),
        "event_type": event_type,
        "event_value": event_value if event_value is not None else {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{participant_id}.jsonl"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_participant_log(participant_id: str) -> list[dict]:
    """Read all events for a participant and return as a list of dicts."""
    log_path = LOGS_DIR / f"{participant_id}.jsonl"
    if not log_path.exists():
        return []
    events = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def compute_derived_metrics(events: list[dict]) -> dict:
    """Compute post-hoc derived metrics from a list of event dicts.

    Returns a flat dictionary of metric names → values.
    """
    metrics: dict = {}

    def count(etype):
        return sum(1 for e in events if e["event_type"] == etype)

    # Efficiency
    task_start_ev = next((e for e in events if e["event_type"] == "task_start"), None)
    task_submit_ev = next((e for e in reversed(events) if e["event_type"] == "task_submit"), None)

    if task_start_ev and task_submit_ev:
        t0 = datetime.fromisoformat(task_start_ev["timestamp"])
        t1 = datetime.fromisoformat(task_submit_ev["timestamp"])
        metrics["task_completion_time_seconds"] = (t1 - t0).total_seconds()
    else:
        metrics["task_completion_time_seconds"] = None

    metrics["num_search_queries"] = count("search_query")
    metrics["num_keyword_refinements"] = count("keyword_refine")
    metrics["num_papers_opened"] = count("paper_open")
    metrics["num_papers_selected"] = count("paper_select")
    metrics["num_deep_research_link_clicks"] = count("deep_research_link_click")

    # Time tracking by condition and section
    time_metrics = {}
    for i in range(len(events) - 1):
        curr_e = events[i]
        next_e = events[i+1]
        t0 = datetime.fromisoformat(curr_e["timestamp"])
        t1 = datetime.fromisoformat(next_e["timestamp"])
        diff_sec = (t1 - t0).total_seconds()
        
        # Cap max idle time tracked to 5 mins
        if diff_sec > 300:
            diff_sec = 300
            
        cond = curr_e.get("condition", "Unknown")
        sec = curr_e.get("section", "Home")
        
        if cond not in time_metrics:
            time_metrics[cond] = {}
        time_metrics[cond][sec] = time_metrics[cond].get(sec, 0.0) + diff_sec
        
    # Make sure default exists even if empty
    if "A (Manual model)" not in time_metrics:
        time_metrics["A (Manual model)"] = {}
    if "B (AI model)" not in time_metrics:
        time_metrics["B (AI model)"] = {}
        
    metrics["time_metrics"] = time_metrics

    # AI reliance ratio
    ai_calls = count("ai_call")
    search_queries = metrics["num_search_queries"]
    total = ai_calls + search_queries
    metrics["ai_calls"] = ai_calls
    metrics["ai_reliance_ratio"] = ai_calls / total if total > 0 else None

    # Exploration depth
    opened = metrics["num_papers_opened"]
    selected = metrics["num_papers_selected"]
    metrics["exploration_depth"] = opened / selected if selected > 0 else None

    # Verification rate
    verifications = count("source_verification_click")
    ai_outputs = count("ai_output_generated")
    metrics["source_verification_clicks"] = verifications
    metrics["ai_outputs_generated"] = ai_outputs
    metrics["verification_rate"] = verifications / ai_outputs if ai_outputs > 0 else None

    # Deep Research Verification Rate
    dr_opens = sum(1 for e in events if e["event_type"] == "source_verification_click" and e.get("event_value", {}).get("source_type") == "deep_research_external_link")
    dr_runs = sum(1 for e in events if e["event_type"] == "ai_output_generated" and e.get("event_value", {}).get("feature") == "deep_research")
    
    metrics["deep_research_runs"] = dr_runs
    metrics["deep_research_verification_rate"] = dr_opens / dr_runs if dr_runs > 0 else None

    # AI feature breakdown
    ai_features: dict[str, int] = {}
    for e in events:
        if e["event_type"] == "ai_call":
            feat = e.get("event_value", {}).get("feature", "unknown")
            ai_features[feat] = ai_features.get(feat, 0) + 1
    metrics["ai_feature_breakdown"] = ai_features

    # Survey scores
    for e in events:
        if e["event_type"] == "survey_response":
            instrument = e.get("event_value", {}).get("instrument", "")
            responses = e.get("event_value", {}).get("responses", {})
            if instrument == "SUS":
                metrics["sus_score"] = _compute_sus_score(responses)
            elif instrument == "NASA_TLX":
                vals = [v for v in responses.values() if isinstance(v, (int, float))]
                metrics["nasa_tlx_mean"] = sum(vals) / len(vals) if vals else None
            elif instrument == "Trust":
                vals = [v for v in responses.values() if isinstance(v, (int, float))]
                metrics["trust_mean"] = sum(vals) / len(vals) if vals else None

    return metrics


def _compute_sus_score(responses: dict) -> float | None:
    """Compute SUS score from a dict like {"Q1": 4, "Q2": 2, ...}.

    SUS scoring: odd items (1,3,5,7,9) subtract 1; even items (2,4,6,8,10)
    subtract from 5. Sum × 2.5 → 0-100 scale.
    """
    try:
        total = 0
        for i in range(1, 11):
            val = responses.get(f"Q{i}")
            if val is None:
                return None
            if i % 2 == 1:  # odd
                total += val - 1
            else:            # even
                total += 5 - val
        return total * 2.5
    except Exception:
        return None


def events_to_csv_string(events: list[dict]) -> str:
    """Convert event list to a CSV string suitable for download."""
    import csv
    import io

    if not events:
        return ""

    buf = io.StringIO()
    fieldnames = [
        "participant_id", "participant_name", "condition", "task_id",
        "event_type", "event_value", "timestamp",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for e in events:
        row = dict(e)
        row["event_value"] = json.dumps(row.get("event_value", {}))
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    return buf.getvalue()
