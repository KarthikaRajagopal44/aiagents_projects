from src.vector_store import VectorStoreManager
from src.ragchain import AdvancedRAGSystem

# Load chunks or documents (assuming you already saved them)
store_manager = VectorStoreManager()
vector_db = store_manager.load_store()

# Optionally create BM25 retriever from chunks (you can load same chunks as used to build vector store)
# For test purposes, let's just load from vector store
# You can also re-load documents from PDF using PyPDFLoader if needed
bm25_retriever = store_manager.get_bm25_retriever([])  # pass chunks list if you have it

# Initialize advanced RAG
rag = AdvancedRAGSystem(vector_db, bm25_retriever)

# Test query
query = "Explain the main topic of my_document.pdf"
rag.ask_with_debug(query)
