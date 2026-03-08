import os
import sys
import time
import warnings
import asyncio
import streamlit as st
import edge_tts
import speech_recognition as sr

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Library Imports
# -----------------------------------------------------------------------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------------------------
# 2. Configuration
# -----------------------------------------------------------------------------
GROQ_API_KEY = "your_api_key_here"

# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------
async def _generate_audio_file(text, output_file):
    """Async helper to generate audio using Edge-TTS."""
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(output_file)

def generate_tts_audio(text):
    """Generates an MP3 file from text for browser playback."""
    start_time = time.time()
    output_file = "response.mp3"
    
    # Generate the file
    asyncio.run(_generate_audio_file(text, output_file))
    
    latency = time.time() - start_time
    return output_file, latency

# -----------------------------------------------------------------------------
# 4. RAG Engine Class
# -----------------------------------------------------------------------------
class RAGReasoningEngine:
    def __init__(self):
        # 1. Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. LLM (Groq)
        self.llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        self.rag_chain = None

    def load_knowledge_base(self):
        faq_data = [
            "Q: What is Machine Learning? A: Machine learning is a subset of AI where computers learn from data.",
            "Q: Who are you? A: I am an AI assistant powered by Groq and LangChain.",
            "Q: How to reset? A: Please restart the device.",
        ]
        return [Document(page_content=text) for text in faq_data]

    def build_vector_index(self):
        documents = self.load_knowledge_base()
        vector_store = FAISS.from_documents(documents, self.embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        
        template = """
        You are a helpful voice assistant. Answer concisely based on context.
        Context: {context}
        Question: {query}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        self.rag_chain = (
            {"context": retriever, "query": RunnablePassthrough()} 
            | prompt 
            | self.llm 
            | StrOutputParser()
        )

    def get_response(self, query):
        return self.rag_chain.invoke(query)

# -----------------------------------------------------------------------------
# 5. Streamlit Application
# -----------------------------------------------------------------------------
# Cache the engine so it loads only once
@st.cache_resource
def init_engine():
    engine = RAGReasoningEngine()
    engine.build_vector_index()
    return engine

def main():
    st.set_page_config(page_title="RAG Voice Assistant", layout="centered")
    st.title("🤖 RAG Voice Assistant")
    st.markdown("Ask questions via text or microphone.")

    # Initialize Engine
    if "engine" not in st.session_state:
        with st.spinner("Loading AI Models... (This may take a minute)"):
            st.session_state.engine = init_engine()

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "audio" in message:
                st.audio(message["audio"], format="audio/mp3")

    # 1. Text Input
    text_input = st.chat_input("Type your question here...")

    # 2. Audio Input (Browser Microphone)
    # Note: This creates a 'Record' button in the UI
    audio_input = st.audio_input("Or record a voice message 🎙️")

    # Determine which input to process
    user_query = None
    if text_input:
        user_query = text_input
    elif audio_input:
        # Process Audio via SpeechRecognition
        with st.spinner("Transcribing audio..."):
            try:
                # Save the uploaded audio to a temp file for the recognizer
                temp_wav = "temp_streamlit_mic.wav"
                with open(temp_wav, "wb") as f:
                    f.write(audio_input.getvalue())
                
                recognizer = sr.Recognizer()
                with sr.AudioFile(temp_wav) as source:
                    audio_data = recognizer.record(source)
                    user_query = recognizer.recognize_google(audio_data)
                    
                st.info(f"You said: **{user_query}**")
            except Exception as e:
                st.error(f"Could not understand audio: {e}")

    # Process the query
    if user_query:
        # Display User Message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = st.session_state.engine.get_response(user_query)
                st.markdown(response_text)
                
                # Generate Voice
                with st.spinner("Generating Voice..."):
                    audio_file, latency = generate_tts_audio(response_text)
                    st.audio(audio_file, format="audio/mp3")
                    st.caption(f"TTS Latency: {latency:.2f}s")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text, 
                "audio": audio_file
            })

if __name__ == "__main__":
    main()