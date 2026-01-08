import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY") # New
    VECTOR_DB_DIR = "db/chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    MODEL_NAME = "gpt-4o-mini"
    
    # Hybrid Search Weights (0.7 Semantic + 0.3 Keyword is usually best)
    SEMANTIC_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3