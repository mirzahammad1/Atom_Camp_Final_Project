"""
utils/ui_helpers.py

Shared UI utilities for both Chatbot and Website pages.
Eliminates duplicated code between 1_Chatbot.py and 2_EDU_Assist_Website.py
"""

from __future__ import annotations

import streamlit as st
from pipeline import get_bot


# =====================================
# Bot Loader (cached across pages)
# =====================================

@st.cache_resource(show_spinner=False)
def load_bot():
    """Load and cache the OrchestratorAgent — runs only once per session."""
    return get_bot()


# =====================================
# Loader Animation CSS
# =====================================

def apply_loader_css() -> None:
    st.markdown(
        """
        <style>
        .init-wrap{
          display:flex; align-items:center; justify-content:center;
          gap:14px; padding:14px 12px; margin: 10px 0 14px 0;
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(20,20,20,0.45);
          backdrop-filter: blur(10px);
        }
        .book { position:relative; width:52px; height:40px; perspective:200px; }
        .page {
          position:absolute; width:26px; height:38px; top:1px; right:0;
          background:rgba(255,255,255,0.16);
          border:1px solid rgba(255,255,255,0.10);
          border-radius:6px;
          transform-origin:left center;
          animation:flip 0.85s infinite ease-in-out;
        }
        .page:nth-child(2){ animation-delay:0.12s; opacity:0.9; }
        .page:nth-child(3){ animation-delay:0.24s; opacity:0.8; }
        .cover {
          position:absolute; left:0; top:0; width:26px; height:40px;
          background:rgba(142,45,226,0.35);
          border:1px solid rgba(142,45,226,0.35);
          border-radius:8px;
        }
        @keyframes flip {
          0%   { transform:rotateY(0deg); }
          50%  { transform:rotateY(-110deg); }
          100% { transform:rotateY(0deg); }
        }

        /* ✅ Verification badge styles */
        .badge-verified {
          display:inline-block; padding:3px 9px; border-radius:999px;
          border:1px solid rgba(74,222,128,0.4);
          background:rgba(74,222,128,0.08);
          color:#4ade80; font-size:11px; margin-right:6px;
        }
        .badge-unverified {
          display:inline-block; padding:3px 9px; border-radius:999px;
          border:1px solid rgba(251,191,36,0.4);
          background:rgba(251,191,36,0.08);
          color:#fbbf24; font-size:11px; margin-right:6px;
        }
        .badge-source {
          display:inline-block; padding:3px 9px; border-radius:999px;
          border:1px solid rgba(255,255,255,0.14);
          background:rgba(255,255,255,0.06);
          font-size:11px; margin-right:6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_init_loader(text: str = "Initializing EDU Assist...") -> None:
    st.markdown(
        f"""
        <div class="init-wrap">
          <div class="book">
            <div class="cover"></div>
            <div class="page"></div>
            <div class="page"></div>
            <div class="page"></div>
          </div>
          <div style="color:#e5e7eb; font-weight:600;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================
# Safe Bot Answer
# =====================================

def safe_answer(bot, prompt: str) -> dict:
    """
    Wraps bot.answer() safely.
    Returns a dict with source, answer, consistency_passed, consistency_attempts.
    """
    try:
        result = bot.answer(prompt)
        return {
            "source": result.get("source", "Unknown"),
            "answer": (result.get("answer", "") or "").strip() or "No answer returned.",
            "consistency_passed": result.get("consistency_passed", "true"),
            "consistency_attempts": result.get("consistency_attempts", "0"),
        }
    except Exception as e:
        return {
            "source": "Error",
            "answer": f"❌ {e}",
            "consistency_passed": "false",
            "consistency_attempts": "0",
        }


# =====================================
# Answer Content Builder
# =====================================

def build_answer_content(out: dict) -> str:
    """
    Builds the HTML content string shown in the chat bubble.
    Includes source badge + ✅/⚠️ verification badge.
    """
    source = out["source"]
    answer = out["answer"]
    passed = out.get("consistency_passed", "true").lower() == "true"
    attempts = out.get("consistency_attempts", "0")

    # Only show verification badge when LLM was involved
    llm_involved = "LLM" in source or "Agent →" in source

    badges = f'<span class="badge-source">📡 {source}</span>'

    if llm_involved:
        if passed:
            badges += '<span class="badge-verified">✅ Verified</span>'
        else:
            badges += f'<span class="badge-unverified">⚠️ Unverified ({attempts} attempts)</span>'

    return f"{badges}\n\n{answer}"


# =====================================
# Context Awareness
# =====================================

_STOP_WORDS = {
    "what", "is", "are", "was", "were", "the", "a", "an", "tell", "me",
    "about", "how", "many", "does", "do", "kiet", "university", "please",
    "can", "you", "i", "want", "know", "give", "show", "explain",
    "describe", "list", "find", "get", "any", "all", "some", "its",
    "their", "there", "which", "who", "when", "where", "will", "would",
    "could", "should", "have", "has", "had", "and", "or", "of", "for",
    "in", "at", "by", "with", "this", "that", "my", "your", "our",
    "also", "more", "from", "not", "but",
}

# Named topics — a question containing one of these stands on its own
_NAMED_TOPICS = {
    "code jung", "hostel", "transport", "scholarship", "sports", "library",
    "lab", "clubs", "alumni", "jobs", "research", "oric", "qec", "lms",
    "portal", "campus", "location", "contact", "vision", "mission",
    "history", "founder", "principal", "rector", "timetable", "result",
    "internship", "semester", "software", "computer", "engineering",
    "management", "business", "science", "technology", "bachelor",
    "master", "phd", "program", "degree", "faculty", "department",
    "structure", "career", "placement", "accreditation", "ranking",
    "fee structure",
}


def _extract_topic(text: str) -> str:
    """Strip stop words from a question to get its core topic keywords."""
    tokens = text.lower().split()
    return " ".join(t for t in tokens if t not in _STOP_WORDS and len(t) > 2)


def _has_named_topic(question: str) -> bool:
    """Returns True if question has a specific named topic — stands alone."""
    q = question.lower()
    return any(topic in q for topic in _NAMED_TOPICS)


def _is_vague(question: str) -> bool:
    """
    Returns True if question is too short/generic to stand alone.

    Vague = no named topic AND 6 words or fewer.

    Examples:
        "what documents are required"  → vague  (4 words, no named topic)
        "when is the last date"        → vague  (5 words, no named topic)
        "tell me about code jung"      → NOT vague  (has named topic)
        "what is the fee structure for software engineering" → NOT vague (7 words)
    """
    if _has_named_topic(question):
        return False
    return len(question.strip().split()) <= 6


def enrich_with_context(current_question: str, messages: list) -> str:
    """
    Enriches a vague follow-up question with the topic from the previous question.

    Examples:
        Previous: "what is the admission process"
        Current:  "what documents are required"
        Enriched: "admission process what documents are required"

        Previous: "tell me about scholarship"
        Current:  "is it free"
        Enriched: "scholarship is it free"

    Rules:
        - Only enriches if current question is vague (<=6 words, no named topic)
        - Looks back at the last user message only
        - Does nothing if no previous messages exist
    """
    if not _is_vague(current_question):
        return current_question

    last_user_msg = None
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    if not last_user_msg:
        return current_question

    topic = _extract_topic(last_user_msg)
    if not topic:
        return current_question

    return f"{topic} {current_question}"


# =====================================
# Session State Init
# =====================================

def init_chat_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


# =====================================
# Message Rendering
# =====================================

def render_messages(empty_msg: str = "👋 Hi! I'm **EDU Assist**. Ask me anything about academics, admissions, policies, or departments.") -> None:
    if not st.session_state.messages:
        st.info(empty_msg)

    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"], unsafe_allow_html=True)


def handle_input(bot) -> None:
    prompt = st.chat_input("Ask EDU Assist...")
    if not prompt:
        return

    # ✅ Store original question as-is for display
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ✅ Enrich query with context from previous question if vague
    # Example: "what documents?" + prev "admission process" → "admission process what documents?"
    enriched_prompt = enrich_with_context(prompt, st.session_state.messages[:-1])

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            out = safe_answer(bot, enriched_prompt)
            content = build_answer_content(out)
            st.session_state.messages.append({"role": "assistant", "content": content})

    st.rerun()