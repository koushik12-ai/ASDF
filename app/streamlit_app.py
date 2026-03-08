# Streamlit Web Application — Voice AI RAG Assistant
# Full-stack UI: text or voice input → RAG engine → TTS audio response.
import os
import time
import warnings
import asyncio
import tempfile

import streamlit as st
import edge_tts
import speech_recognition as sr

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Embeddings import (with fallback for older langchain versions)
# -----------------------------------------------------------------------------
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.warning("Please set your GROQ_API_KEY environment variable.")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
async def _generate_audio_file(text: str, output_file: str):
    """Async helper to generate audio using Edge-TTS."""
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(output_file)


def generate_tts_audio(text: str):
    """Generates an MP3 audio file from text. Returns (file_path, latency_seconds)."""
    start_time = time.time()
    output_file = "response.mp3"
    try:
        try:
            asyncio.get_running_loop()
            new_loop = asyncio.new_event_loop()
            new_loop.run_until_complete(_generate_audio_file(text, output_file))
            new_loop.close()
        except RuntimeError:
            asyncio.run(_generate_audio_file(text, output_file))
    except Exception:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_generate_audio_file(text, output_file))
        loop.close()
    return output_file, time.time() - start_time


def format_docs(docs):
    """Joins retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def transcribe_audio_file(uploaded_audio):
    """Transcribes a Streamlit audio upload to text using Google SpeechRecognition."""
    recognizer = sr.Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_audio.getvalue())
        temp_path = tmp_file.name
    try:
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -----------------------------------------------------------------------------
# RAG Engine
# -----------------------------------------------------------------------------
class RAGReasoningEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.1,
        )
        self.rag_chain = None

    def load_knowledge_base(self):
        faq_data = [
            "Q: What is Machine Learning? A: Machine learning is a subset of AI where computers learn from data.",
            "Q: Who are you? A: I am an AI assistant powered by Groq and LangChain.",
            "Q: How to reset? A: Please restart the device.",
            "Q: What is RAG? A: Retrieval-Augmented Generation combines retrieval from a knowledge base with a language model.",
            "Q: What is ASR? A: ASR stands for Automatic Speech Recognition, which converts spoken audio into text.",
        ]
        return [Document(page_content=text) for text in faq_data]

    def build_vector_index(self):
        documents = self.load_knowledge_base()
        vector_store = FAISS.from_documents(documents, self.embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})

        template = """
You are a helpful voice assistant.
Answer the question concisely using only the provided context.
If the answer is not in the context, say: "I could not find that in the knowledge base."

Context:
{context}

Question:
{query}

Answer:
"""
        prompt = ChatPromptTemplate.from_template(template)
        self.rag_chain = (
            {"context": retriever | RunnableLambda(format_docs), "query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def get_response(self, query: str) -> str:
        if not self.rag_chain:
            raise ValueError("RAG chain not initialized. Call build_vector_index() first.")
        return self.rag_chain.invoke(query)


# -----------------------------------------------------------------------------
# Streamlit Application
# -----------------------------------------------------------------------------
@st.cache_resource
def init_engine():
    engine = RAGReasoningEngine()
    engine.build_vector_index()
    return engine


def main():
    st.set_page_config(page_title="RAG Voice Assistant", layout="centered")
    st.title("🤖 RAG Voice Assistant")
    st.markdown("Ask questions via text or microphone.")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set.")
        st.stop()

    if "engine" not in st.session_state:
        with st.spinner("Loading AI Models..."):
            st.session_state.engine = init_engine()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "audio" in message and message["audio"] and os.path.exists(message["audio"]):
                st.audio(message["audio"], format="audio/mp3")

    text_input = st.chat_input("Type your question here...")
    audio_input = st.audio_input("Or record a voice message 🎙️")
    user_query = None

    if text_input:
        user_query = text_input.strip()
    elif audio_input:
        with st.spinner("Transcribing audio..."):
            try:
                user_query = transcribe_audio_file(audio_input)
                st.info(f"You said: **{user_query}**")
            except Exception as e:
                st.error(f"Could not understand audio: {e}")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        response_text = ""
        audio_file = None
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    response_text = st.session_state.engine.get_response(user_query)
                    st.markdown(response_text)
                with st.spinner("Generating Voice..."):
                    audio_file, latency = generate_tts_audio(response_text)
                    if audio_file and os.path.exists(audio_file):
                        st.audio(audio_file, format="audio/mp3")
                    st.caption(f"TTS Latency: {latency:.2f}s")
            except Exception as e:
                response_text = f"Error generating response: {e}"
                st.error(response_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "audio": audio_file
        })


if __name__ == "__main__":
    main()
