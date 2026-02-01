
# ✈️ AeroCraft ACE-900 GraphRAG Assistant

A sophisticated **Knowledge Graph-based Retrieval-Augmented Generation (GraphRAG)** application designed to provide precise, relationship-aware answers from the AeroCraft ACE-900 Operations Manual.

Unlike standard RAG, which relies solely on semantic similarity, this system extracts entities and their interconnections to build a **Knowledge Graph**, ensuring the AI understands the "chain of events" in complex aviation procedures.

---

## 🏗️ System Architecture

The project follows a **Modular OOP Architecture** consisting of four primary layers:

1.  **Ingestion Layer**: Parses PDF/Text manuals and uses an LLM to perform **Named Entity Recognition (NER)** and **Relationship Extraction**, converting text into `[Subject -> Predicate -> Object]` triples.
2.  **Knowledge Layer**: Stores these triples in a **Directed Graph (NetworkX)**. Nodes represent components/states (e.g., "Fire Handle"), and edges represent actions (e.g., "activates").
3.  **Retrieval Layer**: When a user queries the system, it performs a **Bidirectional Graph Search**. It identifies nodes in the query and retrieves both their "Successors" (what they trigger) and "Predecessors" (what triggers them).
4.  **Generation Layer**: Constrains the LLM (Qwen-2.5-7B) using the retrieved graph context, preventing hallucinations and ensuring answers are grounded strictly in the manual.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit |
| **Orchestration** | LangChain |
| **LLM (Reasoning)** | Qwen-2.5-7B-Instruct (via Hugging Face Inference API) |
| **Embeddings** | Sentence-Transformers (Local: `all-MiniLM-L6-v2`) |
| **Graph Database** | NetworkX (In-memory Directed Graph) |
| **Data Parsing** | PyPDF |
| **Visualization** | Matplotlib |

---

## 🌟 Key Features

*   **Relationship-Aware Retrieval**: Understands complex dependencies (e.g., *Master Switch -> Primary Bus -> Fuel Pump*).
*   **Zero-Hallucination Guardrails**: Uses strict system prompting to force the AI to only answer from the Knowledge Graph.
*   **Real-time Monitoring**: A "System Monitor" sidebar tracks intermediate steps from data ingestion to graph retrieval.
*   **Interactive Visualization**: Renders the Knowledge Graph dynamically as it is built from the document.
*   **Bidirectional Logic**: Can answer both "What happens when I pull X?" and "How do I activate Y?".

---

## 📂 Project Structure

```text
ace900_graph_rag/
├── src/
│   ├── llm_handler.py     # LLM communication & JSON triple extraction
│   ├── graph_manager.py   # NetworkX logic & Graph Visualization
│   ├── orchestrator.py    # Workflow logic & Prompt Engineering
│   └── logger.py          # Streamlit-based system monitoring
├── data/
│   └── manual.pdf         # Source document
├── app.py                 # Main Streamlit UI entry point
├── .env                   # Hugging Face API Key
└── requirements.txt       # Project dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
*   Python 3.9+
*   Hugging Face API Token ([Get one here](https://huggingface.co/settings/tokens))

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/KarthikaRajagopal44/aiagents_projects/tree/main/graph_rag.git
cd graph_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory:
```text
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

### 4. Running the Application
```bash
streamlit run app.py
```

---

## 🔍 Implementation Deep-Dive

### Knowledge Extraction
The system uses **Few-Shot Prompting** to guide the LLM into returning structured JSON. It filters out "Legal Metadata" (Copyrights, disclaimers) to ensure the graph remains focused on operational procedures.

### Graph Retrieval Logic
Standard RAG often fails with questions like "How do I activate the Extinguisher?". 
**Our Fix**: The `GraphManager` implements a `predecessors` search. It finds the "Extinguisher" node and looks *backwards* along the incoming edge to find the "Fire Handle," correctly identifying it as the trigger.

### System Monitoring
The application satisfies technical transparency requirements by logging:
*   **Data Ingestion**: Character counts and file status.
*   **KG Construction**: Number of triples successfully parsed into the graph.
*   **Knowledge Retrieval**: The specific graph facts retrieved before the LLM generates an answer.

---

## 📊 Example Queries & Results

*   **Query**: "What happens when the Fire Handle is pulled?"
    *   **Logic**: Finds "Fire Handle" node -> Follows outgoing edges.
    *   **Result**: 1. Extinguisher Bottle Activation, 2. Fuel Cut-off, 3. Emergency Shutdown.

*   **Query**: "How do I activate the Extinguisher Bottle?"
    *   **Logic**: Finds "Extinguisher Bottle" node -> Follows incoming edges.
    *   **Result**: "Engage the Fire Handle."

---

## 📜 License
This project is developed for educational purposes as part of an Aviation AI Technical Task.

---
**Author**: Karthika Rajagopal 
**Project**: AeroCraft ACE-900 GraphRAG Technical Implementation
