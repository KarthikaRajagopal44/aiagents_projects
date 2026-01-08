# from src.ingestion import DataIngestor
# from src.vector_store import VectorStoreManager
# from src.ragchain import RAGSystem
# import os

# def main():
#     # 1. Setup paths
#     pdf_path = "/home/karthika/Production_RAG/production_rag/data/my_document.pdf" # Place a PDF here!
    
#     if not os.path.exists(pdf_path):
#         print(f"Please place a PDF at {pdf_path}")
#         return

#     # 2. Ingest Data (Only do this once or if data changes)
#     ingestor = DataIngestor(pdf_path)
#     chunks = ingestor.load_and_split()

#     # 3. Store in Vector DB
#     store_manager = VectorStoreManager()
#     vector_db = store_manager.create_store(chunks)

#     # 4. Initialize RAG System
#     rag = RAGSystem(vector_db)

#     # 5. Interactive Loop
#     print("\n--- RAG System Ready. Type 'exit' to quit. ---")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             break
        
#         response = rag.ask(user_input)
#         print(f"\nAI: {response['result']}\n")

# if __name__ == "__main__":
#     main()

# from src.ingestion import DataIngestor
# from src.vector_store import VectorStoreManager
# from src.ragchain import AdvancedRAGSystem
# import os

# def main():
#     pdf_path = "data/my_document.pdf"
    
#     if not os.path.exists(pdf_path):
#         print(f"Please place a PDF at {pdf_path}")
#         return

#     # 1. Ingest Data
#     ingestor = DataIngestor(pdf_path)
#     chunks = ingestor.load_and_split()

#     # 2. Setup Storage
#     store_manager = VectorStoreManager()
#     vector_db = store_manager.create_store(chunks)
#     bm25_retriever = store_manager.get_bm25_retriever(chunks)

#     # 3. Initialize Advanced RAG System
#     rag = AdvancedRAGSystem(vector_db, bm25_retriever)

#     # 4. Interactive Loop
#     print("\n--- Advanced Hybrid RAG System Ready (with Re-ranking) ---")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() == 'exit':
#             break
        
#         result = rag.ask(user_input)
        
#         print(f"\nAI: {result['result']}")
        
#         # PRO TIP: Show the sources to prove the re-ranker worked!
#         print("\n--- Sources Used ---")
#         for i, doc in enumerate(result['source_documents']):
#             print(f"Source {i+1}: {doc.page_content[:100]}...")
#         print("-" * 30)

# if __name__ == "__main__":
#     main()

from src.ingestion import DataIngestor
from src.vector_store import VectorStoreManager
from src.ragchain import AdvancedRAGSystem # To get our retriever
from src.graph import build_graph
from src.evaluator import RAGEvaluator # New
import os

def main():
    pdf_path = "data/my_document.pdf"
    
    # 1. Setup Data & Retriever (From Phase 2)
    ingestor = DataIngestor(pdf_path)
    chunks = ingestor.load_and_split()
    store_manager = VectorStoreManager()
    vector_db = store_manager.create_store(chunks)
    bm25_retriever = store_manager.get_bm25_retriever(chunks)
    
    # Get the combined hybrid + rerank retriever
    advanced_rag = AdvancedRAGSystem(vector_db, bm25_retriever)
    retriever = advanced_rag.rerank_retriever

    # 2. Build the Agentic Graph
    app = build_graph(retriever)

    # 3. Execution
    print("\n--- Agentic RAG Platform Online ---")
    user_query = input("Ask a complex question: ")
    
    inputs = {"question": user_query, "steps": []}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished Node: {key}")

    print("\nFinal Result:")
    # The last node to run is 'generate', so the final output is in the state
    # We can invoke one last time to get the dictionary
    final_state = app.invoke(inputs)
    print(final_state["generation"])

     # 3. RUN EVALUATION (The "Engineer" part)
    print("\n--- Running RAGAS Evaluation ---")
    evaluator = RAGEvaluator()
    eval_results = evaluator.run_evaluation(user_query, answer, docs)
    
    print("\n--- PERFORMANCE METRICS ---")
    # Faithfulness: 1.0 means NO hallucinations.
    # Answer Relevancy: 1.0 means it perfectly answered the prompt.
    print(eval_results[['faithfulness', 'answer_relevancy', 'context_precision']])

if __name__ == "__main__":
    main()