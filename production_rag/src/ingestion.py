from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config

class DataIngestor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_split(self):
        print(f"--- Loading document: {self.file_path} ---")
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(docs)
        print(f"--- Created {len(chunks)} chunks ---")
        return chunks



