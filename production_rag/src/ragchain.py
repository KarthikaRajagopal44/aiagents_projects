from langchain_openai import ChatOpenAI
# # from langchain.chains import RetrievalQA
# from langchain_community.chains import RetrievalQA
from src.config import Config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank




class RAGSystem:
    def __init__(self, vector_db):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0)
        self.retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        
        # This connects the search (Retriever) to the Brain (LLM)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever
        )

    def ask(self, query):
        return self.qa_chain.invoke(query)

class AdvancedRAGSystem:
    def __init__(self, vector_db, bm25_retriever):
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0)
        
        # 1. Setup Semantic Retriever
        vector_retriever = vector_db.as_retriever(search_kwargs={"k": 5})

        # 2. Setup Ensemble (Hybrid) Retriever
        # Combines Chroma + BM25
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[Config.SEMANTIC_WEIGHT, Config.KEYWORD_WEIGHT]
        )

        # 3. Setup Re-ranker (Cohere)
        # This takes the hybrid results and re-orders them perfectly
        compressor = CohereRerank(
            cohere_api_key=Config.COHERE_API_KEY, 
            model="rerank-english-v3.0", 
            top_n=3
        )
        
        self.rerank_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=self.ensemble_retriever
        )

        # 4. Final Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.rerank_retriever,
            return_source_documents=True # Useful for debugging
        )


    def ask(self, query):
        return self.qa_chain.invoke(query)