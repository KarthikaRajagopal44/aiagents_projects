import os
import json
import re
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class LLMHandler:
    def __init__(self):
        self.api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        self.repo_id = "Qwen/Qwen2.5-7B-Instruct"
        
        llm_endpoint = HuggingFaceEndpoint(
            repo_id=self.repo_id,
            huggingfacehub_api_token=self.api_token,
            task="text-generation",
            timeout=300
        )
        self.chat_model = ChatHuggingFace(llm=llm_endpoint)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def extract_triples(self, text):
        # NEW PROMPT: Specifically tells the AI to ignore legal/copyright info
        prompt = f"""Task: Extract technical aviation procedures and system connections from the text below.
        
        CRITICAL RULES:
        1. IGNORE copyright, company names, legal rights, and distribution info.
        2. FOCUS ONLY on triggers (e.g. "If X happens, do Y") and system links.
        3. Format: Return ONLY a JSON list of lists.
        
        Example: [["Fire Handle", "activates", "Extinguisher Bottle"], ["Master Switch", "energizes", "Primary Bus"]]

        Text to analyze:
        {text} 
        """
        
        try:
            response = self.chat_model.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            # Use regex to find the JSON part
            json_match = re.search(r'\[\s*\[.*\]\s*\]', content, re.DOTALL)
            if json_match:
                triples = json.loads(json_match.group())
                # Final check: Remove triples containing legal words
                legal_words = {"copyright", "rights", "reproduction", "distribution", "property"}
                filtered = [t for t in triples if not any(word in str(t).lower() for word in legal_words)]
                return filtered
            return []
        except Exception as e:
            return []
