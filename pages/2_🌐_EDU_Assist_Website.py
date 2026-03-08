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
st.set_page_config(page_title="EDU Assist | Virtual Assistant", page_icon="🌐", layout="wide")


def apply_website_theme() -> None:
    bg = "#0f1117"
    fg = "#ffffff"
    card = "#1a1c23"
    accent = "#8e2de2"

    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {fg}; }}
        [data-testid="stSidebar"] {{ background-color: {card}; }}
        .stChatInput textarea {{ background-color: {card}; color: {fg}; }}

        .hero {{
          background: linear-gradient(135deg,#6a00ff,{accent});
          padding: 18px; border-radius: 16px; margin-bottom: 14px;
        }}
        .hero h2 {{ margin:0; }}
        .hero p {{ margin:6px 0 0 0; opacity:0.9; }}
        .sb-btn button {{ border-radius: 12px !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🤖 EDU Assist")
        st.caption("Intelligent Academic Assistant")
        st.markdown("---")

        st.markdown('<div class="sb-btn">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Back", use_container_width=True, key="web_back_chat"):
                st.switch_page("pages/1_🤖_Chatbot.py")
        with c2:
            if st.button("Clear", use_container_width=True, key="web_clear"):
                st.session_state.messages = []
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 💬 Chat History")

        if not st.session_state.messages:
            st.caption("No conversations yet.")
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    preview = msg["content"][:45]
                    st.markdown(f"- {preview}...")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h2>🎓 Welcome to <b>EDU Assist</b></h2>
          <p>Explore academic programs, admissions, policies, and more through our intelligent assistant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------- App ----------------
init_chat_state()
apply_website_theme()
apply_loader_css()

init_placeholder = st.empty()
with init_placeholder:
    show_init_loader("Initializing EDU Assist (loading knowledge base)...")

bot = load_bot()
init_placeholder.empty()

render_sidebar()
render_hero()
render_messages()
handle_input(bot)