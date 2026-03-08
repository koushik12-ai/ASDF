# TTS modules: converting text into natural speech
import asyncio
import edge_tts
import os
import time

class TTSEngine:
    def __init__(self):
        print("--> Initializing TTS Engine (Edge-TTS)...")
        
        # Voice options
        # 'en-US-AriaNeural' (Female, natural)
        # 'en-US-GuyNeural' (Male, natural)
        self.voice = "en-US-AriaNeural"

    async def _generate_audio_file(self, text, output_file):
        """Async helper to generate the audio file."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

    def generate_audio(self, text, output_filename="response.mp3"):
        """
        Converts text to speech and saves to a file.
        Returns the filename.
        """
        start_time = time.time()
        
        # Run the async generation
        asyncio.run(self._generate_audio_file(text, output_filename))
        
        duration = time.time() - start_time
        print(f"--> TTS Generated in {duration:.2f} seconds.")
        return output_filename

    def speak(self, text):
        """
        Generates audio and plays it using the system default player.
        (Pygame removed to support Python 3.14)
        """
        output_file = "tts_output.mp3"
        
        # 1. Generate the file
        self.generate_audio(text, output_file)
        
        # 2. Play using OS default player (Windows)
        print("--> Playing audio via system player...")
        try:
            os.startfile(output_file)
        except Exception as e:
            print(f"Error playing audio: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    tts = TTSEngine()
    
    sample_text = "I am running on Python 3.14 without Pygame."
    
    print(f"Input: {sample_text}")
    tts.speak(sample_text)