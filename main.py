import time
import numpy as np

# Import all pipeline modules
from pipeline.audio_input import AudioStreamingClient
from pipeline.noise_suppressor import NoiseSuppressor
from pipeline.vad import VoiceActivityDetector
from pipeline.asr import SpeechToTextEngine
from pipeline.intelligence_engine import IntelligenceEngine
from pipeline.tts_engine import TTSEngine


class VoiceAssistant:
    def __init__(self):
        print("=" * 40)
        print("Initializing Voice AI Pipeline...")
        print("=" * 40)

        # Stage 1: Audio Input
        self.audio_input = AudioStreamingClient(rate=16000, frames_per_buffer=480)

        # Stage 2: Noise Suppression
        self.noise_suppressor = NoiseSuppressor(rate=16000)

        # Stage 3: Voice Activity Detection
        self.vad = VoiceActivityDetector(sample_rate=16000)

        # Stage 4: Automatic Speech Recognition
        self.asr = SpeechToTextEngine()

        # Stage 5: LLM + RAG Intelligence
        self.llm = IntelligenceEngine()

        # Stage 6: Text-to-Speech Output
        self.tts = TTSEngine()

        print("=" * 40)
        print("System Ready! Speak into the microphone.")
        print("=" * 40)

    def run(self):
        self.audio_input.start_stream()
        
        try:
            while True:
                # Step 1: Read Audio Chunk
                raw_bytes = self.audio_input.read_frame()
                
                if raw_bytes:
                    # Step 2: Noise Suppression
                    clean_bytes = self.noise_suppressor.process(raw_bytes)
                    
                    # Step 3: Voice Activity Detection
                    is_speech, audio_segment = self.vad.process_frame(clean_bytes)
                    
                    # If a complete sentence is detected
                    if is_speech and audio_segment:
                        print("\n[Main] Speech segment detected. Processing...")
                        
                        # Step 4: ASR (Speech to Text)
                        transcription = self.asr.transcribe(audio_segment)
                        user_text = transcription.get('text', '')
                        
                        if user_text:
                            print(f"[Main] User: {user_text}")
                            
                            # Step 5: LLM + RAG (Reasoning)
                            response = self.llm.get_response(user_text)
                            print(f"[Main] AI: {response}")
                            
                            # Step 6: TTS (Text to Speech)
                            self.tts.speak(response)
                        else:
                            print("[Main] Could not understand audio.")
                            
        except KeyboardInterrupt:
            print("\n[Main] Stopping...")
            self.audio_input.stop_stream()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()