import streamlit as st
from src.orchestrator import RAGOrchestrator

st.set_page_config(page_title="ACE-900 GraphRAG", layout="wide")

# Initialize Orchestrator in session state
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = RAGOrchestrator()
    st.session_state.ingested = False

st.title("✈️ AeroCraft ACE-900 Manual Chat (GraphRAG)")

# Sidebar for Monitoring
with st.sidebar:
    st.header("🔍 System Monitor")
    if st.button("Initialize & Build Graph"):
        with st.spinner("Processing..."):
            st.session_state.orchestrator.ingest_document("data\Manual example - AeroCraft ACE-900 (1).pdf")
            st.session_state.ingested = True
            st.success("Knowledge Graph Built!")

    if st.session_state.ingested:
        st.subheader("Knowledge Graph Visualization")
        fig = st.session_state.orchestrator.graph_manager.visualize()
        if fig: st.pyplot(fig)

    st.subheader("Step Logs")
    for log in st.session_state.orchestrator.monitor.get_logs():
        with st.expander(f"Step: {log['step']}"):
            st.write(log['details'])

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Engine Start or Emergency Shutdown..."):
    if not st.session_state.ingested:
        st.error("Please initialize the system in the sidebar first!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.orchestrator.answer_query(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
