import streamlit as st

class StepMonitor:
    """Handles logging of intermediate RAG steps for the user."""
    def __init__(self):
        if "logs" not in st.session_state:
            st.session_state.logs = []

    def log_step(self, step_name, details):
        log_entry = {"step": step_name, "details": details}
        st.session_state.logs.append(log_entry)
        print(f"[{step_name}]: {details}")

    def get_logs(self):
        return st.session_state.logs
