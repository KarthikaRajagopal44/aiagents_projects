import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from src.config import Config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class RAGEvaluator:
    def __init__(self):
        # Ragas uses LLMs to "judge" the quality of the answer
        self.judgement_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings()

    def run_evaluation(self, question, answer, contexts):
        """
        Evaluates a single interaction.
        In a real production app, you'd run this on a 'Golden Dataset' of 20+ questions.
        """
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [[c.page_content for c in contexts]],
        }
        
        dataset = Dataset.from_dict(data)
        
        # We use metrics that don't strictly require 'ground truth' for real-time monitoring
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=self.judgement_llm,
            embeddings=self.embeddings
        )
        
        return result.to_pandas()