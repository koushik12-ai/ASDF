# Module 5: Intelligence Engine (LLM + RAG)
# Connects to the Groq Cloud API with a lightweight local RAG layer.
import openai

# ==========================================
# CONFIGURATION
# ==========================================
# Load API key from environment variable (set GROQ_API_KEY in your .env or shell)
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
# ==========================================


class IntelligenceEngine:
    def __init__(self):
        print("[Intelligence] Initializing Reasoning Layer...")

        # Knowledge base for RAG retrieval
        self.knowledge_base = [
            "The user's name is Koushik.",
            "This is a Voice AI Pipeline project.",
            "The project uses Python 3.14 and Groq API.",
            "The system consists of 6 modules.",
        ]

        # Groq-compatible OpenAI client
        self.client = openai.OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = "llama-3.1-8b-instant"
        print(f"[Intelligence] Connected to Model: {self.model_name}")
        print("[Intelligence] System Ready.")

    def _retrieve_context(self, query: str) -> str:
        """RAG: Keyword-based retrieval from the knowledge base."""
        query_words = set(query.lower().split())
        relevant_docs = [
            doc for doc in self.knowledge_base
            if any(word in doc.lower() for word in query_words)
        ]
        return "\n".join(relevant_docs) if relevant_docs else "No specific context found."

    def get_response(self, user_query: str) -> str:
        """
        Retrieves context and generates a response via the Groq LLM.

        :param user_query: The user's input text.
        :returns: AI-generated response string.
        """
        context = self._retrieve_context(user_query)
        print(f"[Intelligence] Retrieved Context: {context}")

        system_prompt = (
            "You are a helpful assistant. "
            "Answer the user's question using ONLY the context provided below. "
            "If the answer is in the context, state it clearly.\n\n"
            f"Context:\n{context}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Intelligence] API Error: {e}")
            return "Error processing request."


# --- Unit Test ---
if __name__ == "__main__":
    brain = IntelligenceEngine()
    print("\nQuery: What is my name?")
    ans = brain.get_response("What is my name?")
    print(f"AI Response: {ans}")
