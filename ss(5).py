import os
import sys
import time
import warnings
import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import numpy as np
import wave

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
GROQ_API_KEY = "your_api_key_here" # REPLACE WITH YOUR KEY

# Use the newer import to stop the deprecation warning
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

class RAGReasoningEngine:
    def __init__(self):
        print("--> Initializing Bot Systems...")
        
        # 1. Text-to-Speech Engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 190) # Slightly faster speech
        
        # 2. Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 3. LLM (Groq)
        self.llm = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        self.rag_chain = None

    # -------------------------------------------------------------------------
    # Audio Layer (Fixed for Compatibility)
    # -------------------------------------------------------------------------
    def listen_to_audio(self):
        print("   [Listening...] Speak now (4s)")
        
        fs = 44100  # Sample rate
        seconds = 4 # Duration
        
        try:
            # 1. Record audio (This comes as Float32)
            recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished
            
            # 2. Convert Float32 to Int16 (Required for SpeechRecognition)
            # This step fixes the "PCM WAV" error
            recording_int16 = (recording * 32767).astype(np.int16)
            
            # 3. Save as proper WAV file using wave module
            temp_filename = "temp_mic_input.wav"
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 2 bytes = 16-bit
                wf.setframerate(fs)
                wf.writeframes(recording_int16.tobytes())
            
            print("   [Processing Speech...]")
            
            # 4. Recognize
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_filename) as source:
                audio_data = recognizer.record(source)
                
                try:
                    text = recognizer.recognize_google(audio_data)
                    return text
                except sr.UnknownValueError:
                    print("   [Could not understand audio]")
                    return None
                except sr.RequestError:
                    print("   [API Error]")
                    return None
                    
        except Exception as e:
            print(f"   [Mic Error: {e}]")
            return None

    def speak_response(self, text):
        """Converts text to speech."""
        start_time = time.time()
        
        # Speak the text
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        
        latency = time.time() - start_time
        print(f"   [TTS Latency: {latency:.2f}s]")

    # -------------------------------------------------------------------------
    # Knowledge & RAG Logic
    # -------------------------------------------------------------------------
    def load_knowledge_base(self):
        # Custom knowledge
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

    def process_request(self, user_query):
        print("Bot: Thinking...")
        response = self.rag_chain.invoke(user_query)
        print(f"Bot: {response}")
        self.speak_response(response)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if "gsk_" not in GROQ_API_KEY:
        print("ERROR: Paste your Groq API Key!")
    else:
        engine = RAGReasoningEngine()
        engine.build_vector_index()
        
        print("\n" + "="*50)
        print("BOT IS READY. SPEAK TO INTERACT.")
        print("="*50)
        
        while True:
            user_speech = engine.listen_to_audio()
            if user_speech:
                print(f"User: {user_speech}")
                engine.process_request(user_speech)
                print("-" * 40)