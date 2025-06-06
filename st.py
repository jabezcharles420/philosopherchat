import streamlit as st
from app import start_chat, PHILOSOPHER_NAMES
import uuid

st.set_page_config(page_title="Chat with a Philosopher", layout="wide")

st.markdown(
    """
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .chat-bubble-user {
            background-color: #1a1a1a;
            color: #e0e0e0;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: right;
        }
        .chat-bubble-ai {
            background-color: #333333;
            color: #f0f0f0;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">🤔 Chat with a Philosopher</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Choose Your Philosopher")
    philosopher_key = st.selectbox(
        "Philosopher", options=list(PHILOSOPHER_NAMES.keys()), format_func=lambda x: PHILOSOPHER_NAMES[x]
    )

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if st.button("🧹 Start New Conversation"):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())

# Initialize chat history if missing
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Handle input first
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner(f"{PHILOSOPHER_NAMES[philosopher_key]} is thinking..."):
        response, thread_id = start_chat(philosopher_key, user_input, st.session_state.thread_id)

    # Save AI response
    st.session_state.chat_history.append({"role": "ai", "content": response})

# Now display conversation (after updating chat history)
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-ai"><b>{PHILOSOPHER_NAMES[philosopher_key]}:</b> {msg["content"]}</div>', unsafe_allow_html=True)
