import streamlit as st
import requests
import time

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/chat"
st.set_page_config(page_title="HR Policy Assistant", page_icon="🤖", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # session_id matches the user_id in FastAPI request model
    if "user_id" not in st.session_state:
        st.session_state.user_id = "emp_001" # Default or dynamic
    st.session_state.initialized = False

# --- SIDEBAR ---
with st.sidebar:
    st.title("👤 Session Settings")
    st.session_state.user_id = st.text_input("Employee ID", value=st.session_state.user_id)
    
    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.initialized = False
        st.rerun()

# --- MAIN INTERFACE ---
st.title("🤖 HR Policy Assistant")
st.caption("I am your automated HR officer. Please log in with your department to begin.")

# 1. INITIAL HANDSHAKE (Trigger Proactive Greeting)
# This mimics the main.py greeting block by sending an empty string to the API
if not st.session_state.initialized:
    try:
        init_payload = {"question": "", "user_id": st.session_state.user_id}
        response = requests.post(API_URL, json=init_payload)
        if response.status_code == 200:
            data = response.json()
            st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
            st.session_state.initialized = True
    except Exception:
        st.error("⚠️ Backend connection failed. Please ensure FastAPI is running on port 8000.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. CHAT INPUT
if prompt := st.chat_input("Enter your department or query here..."):
    # Add user message to UI and state
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call FastAPI Endpoint
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        full_response = ""
        
        try:
            payload = {
                "question": prompt,
                "user_id": st.session_state.user_id
            }
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                api_data = response.json()
                answer = api_data["answer"]
                is_locked = api_data["requires_department"]
                
                # Typing effect
                for chunk in answer.split(" "):
                    full_response += chunk + " "
                    msg_placeholder.markdown(full_response + "▌")
                    time.sleep(0.04)
                msg_placeholder.markdown(full_response)

                # Visual Hint if still needing department
                if is_locked:
                    st.info("💡 Hint: Just type your department name (e.g., 'Admin' or 'IT').")

                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Connection Error: {e}")