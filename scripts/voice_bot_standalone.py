# Voice Bot — Standalone Script (pyttsx3 + SpeechRecognition)
# A self-contained voice assistant that uses local TTS (pyttsx3) and Google ASR.
# This is a standalone alternative to the full modular pipeline (main.py).
import os
import time
import warnings
import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import numpy as np
import wave

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

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


class VoiceBotStandalone:
    def __init__(self):
        print("--> Initializing Standalone Voice Bot...")

        # Local text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 190)

        # LangChain components
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

    # -------------------------------------------------------------------------
    # Audio: Listen and Speak
    # -------------------------------------------------------------------------
    def listen_to_audio(self):
        """Records 4 seconds of microphone audio and transcribes it to text."""
        print("   [Listening...] Speak now (4s)")
        fs = 44100
        seconds = 4
        try:
            recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            recording_int16 = (recording * 32767).astype(np.int16)
            temp_filename = "temp_mic_input.wav"
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(fs)
                wf.writeframes(recording_int16.tobytes())
            print("   [Processing Speech...]")
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_filename) as source:
                audio_data = recognizer.record(source)
                try:
                    return recognizer.recognize_google(audio_data)
                except sr.UnknownValueError:
                    print("   [Could not understand audio]")
                    return None
                except sr.RequestError:
                    print("   [API Error - check internet connection]")
                    return None
        except Exception as e:
            print(f"   [Mic Error: {e}]")
            return None

    def speak_response(self, text: str):
        """Speaks the given text using the local pyttsx3 TTS engine."""
        start_time = time.time()
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        print(f"   [TTS Latency: {time.time() - start_time:.2f}s]")

    # -------------------------------------------------------------------------
    # RAG: Knowledge Base and Chain
    # -------------------------------------------------------------------------
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

    def process_request(self, user_query: str):
        """Generates and speaks a response to the user query."""
        print("Bot: Thinking...")
        response = self.rag_chain.invoke(user_query)
        print(f"Bot: {response}")
        self.speak_response(response)


# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if "gsk_" not in GROQ_API_KEY:
        print("ERROR: Set your GROQ_API_KEY environment variable.")
    else:
        bot = VoiceBotStandalone()
        bot.build_vector_index()
        print("\n" + "=" * 50)
        print("BOT IS READY. SPEAK TO INTERACT. (Ctrl+C to quit)")
        print("=" * 50)
        while True:
            user_speech = bot.listen_to_audio()
            if user_speech:
                print(f"User: {user_speech}")
                bot.process_request(user_speech)
                print("-" * 40)
