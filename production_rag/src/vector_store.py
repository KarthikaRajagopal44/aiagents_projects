# from langchain_openai import OpenAIEmbeddings

# from src.config import Config

# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.retrievers import BM25Retriever
# from langchain_community.vectorstores import Chroma



# # ------------------- PATCH -------------------
# # Subclass OpenAIEmbeddings to remove 'proxies'
# class FixedOpenAIEmbeddings(OpenAIEmbeddings):
#     def __init__(self, *args, **kwargs):
#         if 'proxies' in kwargs:
#             kwargs.pop('proxies')  # Remove proxies to avoid error
#         super().__init__(*args, **kwargs)
# # ---------------------------------------------

# class VectorStoreManager:
#     def __init__(self):
#         # Use the fixed embeddings class
#         self.embeddings = FixedOpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)

#     def create_store(self, chunks):
#         print("--- Saving chunks to Vector Database ---")
#         vector_db = Chroma.from_documents(
#             # documents=chunks,
#             # embedding=self.embeddings,
#             # persist_directory=Config.VECTOR_DB_DIR
#              documents=chunks,
#              embedding=self.embeddings,
#              persist_directory="chroma_db",  # <-- this is the folder to delete
#              collection_name="my_collection"
#         )
#         return vector_db

#     def load_store(self):
#         return Chroma(
#             persist_directory=Config.VECTOR_DB_DIR,
#             embedding_function=self.embeddings
#         )

#     def get_bm25_retriever(self, chunks):
#         # 2. Keyword Store (BM25)
#         # BM25 is usually kept in memory or saved as a pickle file
#         bm25_retriever = BM25Retriever.from_documents(chunks)
#         bm25_retriever.k = 5 # Retrieve top 5 keyword matches
#         return bm25_retriever

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from src.config import Config


class VectorStoreManager:
    def __init__(self):
        # ✅ FREE, LOCAL, INTERVIEW-SAFE embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def create_store(self, chunks):
        print("--- Saving chunks to Vector Database ---")

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=Config.VECTOR_DB_DIR,
            collection_name="my_collection"
        )

        return vector_db

    def load_store(self):
        return Chroma(
            persist_directory=Config.VECTOR_DB_DIR,
            embedding_function=self.embeddings,
            collection_name="my_collection"
        )

    def get_bm25_retriever(self, chunks):
        # 🔹 Keyword-based retrieval (BM25)
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5
        return bm25_retriever


