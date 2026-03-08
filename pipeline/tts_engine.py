# Module 6: Text-to-Speech (TTS) Engine
# Converts AI text responses to natural-sounding MP3 audio using Edge-TTS.
import asyncio
import os
import time
import edge_tts


class TTSEngine:
    def __init__(self):
        print("--> Initializing TTS Engine (Edge-TTS)...")
        # Voice options:
        #   'en-US-AriaNeural'  - Female, natural
        #   'en-US-GuyNeural'   - Male, natural
        self.voice = "en-US-AriaNeural"

    async def _generate_audio_file(self, text: str, output_file: str):
        """Async helper: generates and saves the audio file."""
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(output_file)

    def generate_audio(self, text: str, output_filename: str = "response.mp3") -> str:
        """
        Converts text to speech and saves to an MP3 file.

        :param text: The text to synthesize.
        :param output_filename: Output file path (default: response.mp3).
        :returns: Path to the generated audio file.
        """
        start_time = time.time()
        asyncio.run(self._generate_audio_file(text, output_filename))
        duration = time.time() - start_time
        print(f"--> TTS Generated in {duration:.2f} seconds.")
        return output_filename

    def speak(self, text: str):
        """
        Generates audio from text and plays it using the OS default player.

        :param text: The text to speak.
        """
        output_file = "tts_output.mp3"
        self.generate_audio(text, output_file)
        print("--> Playing audio via system player...")
        try:
            os.startfile(output_file)
        except Exception as e:
            print(f"Error playing audio: {e}")


# --- Unit Test ---
if __name__ == "__main__":
    tts = TTSEngine()
    sample_text = "Voice AI Pipeline is online and ready."
    print(f"Input: {sample_text}")
    tts.speak(sample_text)
