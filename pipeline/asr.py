# Module 4: Automatic Speech Recognition (ASR)
# Placeholder for the ASR engine that converts audio bytes to text.
#
# Recommended integrations:
#   - OpenAI Whisper (local):  pip install openai-whisper
#   - Groq Whisper (cloud):    via groq Python SDK
#   - SpeechRecognition:       pip install SpeechRecognition
#
# The class below provides the expected interface so main.py can call it.

class SpeechToTextEngine:
    def __init__(self):
        """Initialize the ASR engine."""
        print("[ASR] SpeechToTextEngine initialized (stub).")
        print("[ASR] Replace this stub with a real ASR backend (e.g. Whisper).")

    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Transcribe raw audio bytes into text.

        :param audio_bytes: PCM audio data as bytes.
        :returns: dict with key 'text' containing the transcription string.
        """
        # TODO: Implement actual ASR here.
        # Example with Whisper:
        #   import whisper
        #   model = whisper.load_model("base")
        #   result = model.transcribe(audio_bytes)
        #   return result
        raise NotImplementedError(
            "ASR engine not implemented. "
            "Plug in a Whisper or SpeechRecognition backend."
        )
