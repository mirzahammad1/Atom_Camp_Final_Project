import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(page_title="EDU Assist", page_icon="🤖", layout="centered")


# ---------------- UI Theme ----------------
def apply_home_theme() -> None:
    st.markdown(
        """
        <style>
        body { background: radial-gradient(circle at top, #121212, #0b0d13); }
        .block-container { padding-top: 3rem; }

        .hero { text-align:center; margin-top: 12vh; }
        .hero h1 { font-size: 3rem; font-weight: 800; margin-bottom: 0.25rem; color: #f3f4f6; }
        .hero p  { color: #9ca3af; font-size: 18px; margin-top: 0.6rem; }

        .start-btn button {
            font-size: 56px !important;
            height: 96px !important;
            width: 96px !important;
            border-radius: 999px !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            background: rgba(142,45,226,0.12) !important;
        }
        .start-btn button:hover {
            transform: translateY(-1px);
            transition: 0.15s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    apply_home_theme()

    st.markdown(
        """
        <div class="hero">
          <h1>Welcome to <span style="color:#8e2de2;">EDU Assist</span></h1>
          <p>Your intelligent academic assistant</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.markdown('<div class="start-btn">', unsafe_allow_html=True)
        if st.button("🤖", use_container_width=True, key="home_start_chat"):
            st.switch_page("pages/1_🤖_Chatbot.py")
        st.markdown("</div>", unsafe_allow_html=True)


render_home()