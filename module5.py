import openai

# ==========================================
# CONFIGURATION
# ==========================================
GROQ_API_KEY = "your_api_key_here" 
# ==========================================

class IntelligenceEngine:
    def __init__(self):
        print("[Intelligence] Initializing Reasoning Layer...")
        
        # 1. Initialize Knowledge Base
        self.knowledge_base = [
            "The user's name is Koushik.",
            "This is a Voice AI Pipeline project.",
            "The project uses Python 3.14 and Groq API.",
            "The system consists of 6 modules."
        ]

        # 2. Initialize Client
        self.client = openai.OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = "llama-3.1-8b-instant"
        print(f"[Intelligence] Connected to Model: {self.model_name}")
        print("[Intelligence] System Ready.")

    def _retrieve_context(self, query):
        """RAG: Find relevant documents."""
        query_words = set(query.lower().split())
        relevant_docs = []
        
        for doc in self.knowledge_base:
            # Check if any word from the query appears in the document
            if any(word in doc.lower() for word in query_words):
                relevant_docs.append(doc)
        
        return "\n".join(relevant_docs) if relevant_docs else "No specific context found."

    def get_response(self, user_query):
        # Step A: Retrieve Context
        context = self._retrieve_context(user_query)
        # Debug print to show we found the info
        print(f"[Debug] Retrieved Context: {context}") 

        # Step B: Strong Prompt
        system_prompt = f"""You are a helpful assistant. 
You must answer the user's question using ONLY the context provided below. 
If the answer is in the context, state it clearly. 
Do not say you don't know if the answer is in the context.

Context:
{context}"""

        # Step C: Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1, # Lower temperature = more factual
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[Intelligence] API Error: {e}")
            return "Error processing request."

if __name__ == "__main__":
    brain = IntelligenceEngine()
    print("\nQuery: What is my name?")
    ans = brain.get_response("What is my name?")
    print(f"AI Response: {ans}")