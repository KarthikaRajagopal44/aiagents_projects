from src.config import Config
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.state import AgentState


class Nodes:
    def __init__(self, retriever):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=Config.MODEL_NAME, temperature=0)

    def retrieve(self, state: AgentState):
        """Step 1: Retrieve documents using the Phase 2 Hybrid Retriever"""
        print("--- NODE: RETRIEVING DOCUMENTS ---")
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question, "steps": ["retrieve"]}

    def grade_documents(self, state: AgentState):
        """Step 2: Check if retrieved documents are relevant to the question"""
        print("--- NODE: CHECKING RELEVANCE ---")
        question = state["question"]
        documents = state["documents"]
        
        # Simple LLM grader logic
        prompt = PromptTemplate(
            template="""You are a grader. Is this document relevant to the user question? 
            Answer with a JSON: {{"score": "yes"}} or {{"score": "no"}}
            Question: {question} \n Document: {doc}""",
            input_variables=["question", "doc"],
        )
        
        grader = prompt | self.llm | JsonOutputParser()
        
        filtered_docs = []
        search_again = "no"
        
        for d in documents:
            res = grader.invoke({"question": question, "doc": d.page_content})
            if res["score"] == "yes":
                filtered_docs.append(d)
        
        if not filtered_docs:
            search_again = "yes" # Flag to trigger self-correction
            
        return {"documents": filtered_docs, "question": question, "steps": ["grade"], "search_again": search_again}

    def generate(self, state: AgentState):
        """Step 3: Final Answer Generation"""
        print("--- NODE: GENERATING FINAL ANSWER ---")
        question = state["question"]
        documents = state["documents"]
        
        context = "\n".join([d.page_content for d in documents])
        prompt = f"Answer the question: {question} using only this context: {context}"
        
        response = self.llm.invoke(prompt)
        return {"generation": response.content, "steps": ["generate"]}

    def transform_query(self, state: AgentState):
        """Step 4: Self-Correction (Query Rewriting)"""
        print("--- NODE: REWRITING QUERY ---")
        question = state["question"]
        
        prompt = f"The previous search for '{question}' failed. Re-write this query to be more specific for a search engine."
        better_question = self.llm.invoke(prompt).content
        
        return {"question": better_question, "steps": ["transform_query"]}