import streamlit as st
from utils.ui_helpers import (
    load_bot,
    apply_loader_css,
    show_init_loader,
    init_chat_state,
    render_messages,
    handle_input,
)

# ---------------- Page Config ----------------
st.set_page_config(page_title="EDU Assist Chat", page_icon="🤖", layout="centered")


# ---------------- Theme ----------------
def apply_chat_theme() -> None:
    st.markdown(
        """
        <style>
        body { background: radial-gradient(circle at top, #121212, #0b0d13); }
        .block-container { max-width: 780px; padding-top: 1.25rem; padding-bottom: 3rem; }
        .ctrl button {
          border-radius: 12px !important;
          height: 44px !important;
          font-weight: 700 !important;
          border: 1px solid rgba(255,255,255,0.12) !important;
          background: rgba(255,255,255,0.06) !important;
        }
        .ctrl button:hover {
          background: rgba(255,255,255,0.10) !important;
          transition: 0.12s ease;
        }
        .header {
          display:flex; justify-content:space-between; align-items:center;
          padding: 10px 0 14px 0;
          border-bottom: 1px solid rgba(255,255,255,0.08);
          margin-bottom: 14px;
          font-size: 18px; font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_controls() -> None:
    c1, c2, c3 = st.columns([1, 1, 1], gap="small")
    with c1:
        st.markdown('<div class="ctrl">', unsafe_allow_html=True)
        if st.button("← Back to Home", use_container_width=True, key="chat_back_home"):
            st.switch_page("Home.py")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="ctrl">', unsafe_allow_html=True)
        if st.button("🌐 Website View", use_container_width=True, key="chat_website_view"):
            st.switch_page("pages/2_🌐_EDU_Assist_Website.py")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="ctrl">', unsafe_allow_html=True)
        if st.button("🧹 Clear Chat", use_container_width=True, key="chat_clear"):
            st.session_state.messages = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="header">
          <div>EDU Assist <small style="opacity:0.5;">• online</small></div>
          <div>🤖</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- App ----------------
apply_chat_theme()
apply_loader_css()
init_chat_state()

init_placeholder = st.empty()
with init_placeholder:
    show_init_loader("Initializing EDU Assist (loading knowledge base)...")

bot = load_bot()
init_placeholder.empty()

render_controls()
render_header()
render_messages()
handle_input(bot)