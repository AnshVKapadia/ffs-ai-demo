# main.py
import os
import streamlit as st

# Local modules
from default_bot import generate_default_response
from scholarship_bot import generate_scholarship_response

# --------------------------
# Load API key (Secrets > env)
# --------------------------
def ensure_openai_key():
    """
    Prefer Streamlit Secrets. Fall back to env var for local dev.
    Also writes the key into os.environ so downstream modules work.
    """
    secret_key = st.secrets.get("OPENAI_API_KEY", None)
    env_key = os.getenv("OPENAI_API_KEY")
    key = secret_key or env_key
    if key and env_key != key:
        os.environ["OPENAI_API_KEY"] = key
    source = "Streamlit Secrets" if secret_key else ("Environment variable" if env_key else None)
    return bool(key), source

api_ok, api_source = ensure_openai_key()

# --------------------------
# Streamlit page config
# --------------------------
st.set_page_config(page_title="FFS Multi-Mode Chat", page_icon="🎓", layout="wide")
st.title("🎓 FFS Multi-Mode Chat")
st.caption("Toggle between a Default Tutor/Counselor assistant and a Scholarship Finder.")

# --------------------------
# Sidebar controls
# --------------------------
with st.sidebar:
    st.header("Mode & Settings")
    mode = st.radio("Mode", ["Default Chatbot", "Scholarship Finder"], horizontal=False)

    st.markdown(f"**API Key loaded:** {'✅' if api_ok else '❌'}"
                + (f" ({api_source})" if api_source else ""))

    if not api_ok:
        st.info("Add OPENAI_API_KEY via App → ⋯ → Settings → Secrets (recommended), "
                "or set a local environment variable for development.")

    keep_history = st.checkbox("Remember short chat history", value=True)
    show_usage = st.checkbox("Show token usage (if returned)", value=True)

    if mode == "Default Chatbot":
        # Fixed model: gpt-4o-mini (no dropdown)
        st.write("**Model:** gpt-4o-mini")
        temperature = 0.3
    else:
        # Fixed model: gpt-4o-mini-search-preview (no temperature control)
        st.write("**Model:** gpt-4o-mini-search-preview")
        temperature = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear current chat"):
            if mode == "Default Chatbot":
                st.session_state["history_default"] = []
            else:
                st.session_state["history_scholarship"] = []
            st.rerun()
    with col2:
        if st.button("Clear ALL"):
            st.session_state.clear()
            st.rerun()

# --------------------------
# Session state setup
# --------------------------
if "history_default" not in st.session_state:
    st.session_state.history_default = []  # list[{"role": "user"/"assistant", "content": str}]
if "history_scholarship" not in st.session_state:
    st.session_state.history_scholarship = []

history_key = "history_default" if mode == "Default Chatbot" else "history_scholarship"
history = st.session_state[history_key]

# --------------------------
# Show chat history
# --------------------------
for msg in history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------
# Chat input
# --------------------------
placeholder = (
    "Ask anything—AP/STEM help, study plans, college planning…"
    if mode == "Default Chatbot"
    else "Ask for scholarships (e.g., 'women STEM scholarships for high school seniors')"
)
user_input = st.chat_input(placeholder)

if user_input:
    history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if mode == "Default Chatbot":
        # Fixed model: gpt-4o-mini
        result = generate_default_response(
            user_input=user_input,
            chat_history=history[:-1],      # exclude just-added user
            keep_history=keep_history,
            model="gpt-4o-mini",
            temperature=temperature if temperature is not None else 0.3,
        )
        text = result["text"]
        usage = result["usage"]

        with st.chat_message("assistant"):
            st.markdown(text)
            if show_usage and usage:
                u = usage
                st.caption(
                    f"Tokens — prompt: {u['prompt_tokens']} • completion: {u['completion_tokens']} • total: {u['total_tokens']}"
                )

        history.append({"role": "assistant", "content": text})

    else:
        # Fixed model: gpt-4o-mini-search-preview
        result = generate_scholarship_response(
            user_input=user_input,
            chat_history=history[:-1],
            keep_history=keep_history,
        )
        text = result["clean_text"]
        usage = result["usage"]
        removed = result["flagged_chunks"]

        with st.chat_message("assistant"):
            st.markdown(text or "[No content returned]")
            if removed:
                st.info(f"Removed {len(removed)} expired scholarship(s).")
            if show_usage and usage:
                u = usage
                st.caption(
                    f"Tokens — prompt: {u['prompt_tokens']} • completion: {u['completion_tokens']} • total: {u['total_tokens']}"
                )

        history.append({"role": "assistant", "content": text})

    st.session_state[history_key] = history
