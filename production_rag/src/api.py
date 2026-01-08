from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from src.ingestion import DataIngestor
from src.vector_store import VectorStoreManager
from src.rag_chain import AdvancedRAGSystem
from src.graph import build_graph
from src.evaluator import RAGEvaluator
import os
import shutil

app = FastAPI(title="Agentic RAG Production API")

# Initialize components (In production, use Dependency Injection)
store_manager = VectorStoreManager()
evaluator = RAGEvaluator()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    steps: list
    faithfulness: float
    relevance: float

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload a PDF and index it into the Vector DB"""
    temp_path = f"data/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the document
    ingestor = DataIngestor(temp_path)
    chunks = ingestor.load_and_split()
    store_manager.create_store(chunks)
    
    return {"message": f"Document {file.filename} indexed successfully", "chunks": len(chunks)}

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question to the Agentic RAG system"""
    # 1. Setup the Graph (Phase 3 logic)
    # Note: In a real app, you'd cache the retriever
    vector_db = store_manager.load_vector_store()
    # For BM25, we'd ideally load a saved index, but here we rebuild for simplicity
    # (In high-scale, you'd use a DB like Qdrant that handles Hybrid natively)
    
    # Re-using our Phase 2/3 logic
    # We assume documents were already ingested via /ingest
    advanced_rag = AdvancedRAGSystem(vector_db, None) # Logic update needed for production storage
    agent_app = build_graph(advanced_rag.rerank_retriever)
    
    # 2. Execute Agent
    inputs = {"question": request.question, "steps": []}
    result = agent_app.invoke(inputs)
    
    # 3. Evaluate (Phase 4 logic)
    eval_metrics = evaluator.run_evaluation(
        request.question, 
        result["generation"], 
        result["documents"]
    )
    
    return {
        "answer": result["generation"],
        "steps": result["steps"],
        "faithfulness": float(eval_metrics['faithfulness'].iloc[0]),
        "relevance": float(eval_metrics['answer_relevancy'].iloc[0])
    }