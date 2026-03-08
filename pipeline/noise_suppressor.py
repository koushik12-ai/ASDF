# Module 2: Noise Suppressor
# Applies a NumPy-based amplitude noise gate to clean raw audio chunks.
import numpy as np


class NoiseSuppressor:
    def __init__(self, rate=16000, threshold_db=-40):
        """
        Initializes a simple noise gate using only NumPy.

        :param rate: Sample rate in Hz.
        :param threshold_db: Volume threshold in dB.
                             Sounds below this level are silenced.
                             -40dB suits quiet rooms; -30dB suits noisier environments.
        """
        self.rate = rate
        # Convert dB threshold to linear amplitude (0.0 to 1.0)
        self.threshold = 10.0 ** (threshold_db / 20.0)

    def process(self, audio_chunk_bytes):
        """
        Applies noise gating to an audio chunk.
        If the signal amplitude is below threshold, the chunk is silenced.

        :param audio_chunk_bytes: Raw audio bytes (int16 format).
        :returns: Processed audio as bytes.
        """
        # 1. Convert bytes to Int16 array
        audio_np = np.frombuffer(audio_chunk_bytes, dtype=np.int16)

        # 2. Normalize to -1.0 to 1.0 range
        audio_float = audio_np.astype(np.float32) / 32768.0

        # 3. Calculate peak amplitude
        current_amplitude = np.max(np.abs(audio_float))

        # 4. Apply noise gate
        if current_amplitude < self.threshold:
            processed_audio = np.zeros_like(audio_float)  # Silence noise
        else:
            processed_audio = audio_float  # Keep speech

        # 5. Convert back to Int16 bytes
        processed_audio_int = (processed_audio * 32768.0).astype(np.int16)
        return processed_audio_int.tobytes()


# --- Unit Test ---
if __name__ == "__main__":
    print("Testing NumPy Noise Suppressor...")
    silence = np.zeros(1000, dtype=np.int16)
    speech = np.random.randint(-10000, 10000, size=1000, dtype=np.int16)
    suppressor = NoiseSuppressor(threshold_db=-35)
    out_silence = suppressor.process(silence.tobytes())
    print(f"Silence Output Max: {np.max(np.frombuffer(out_silence, dtype=np.int16))} (Should be 0)")
    out_speech = suppressor.process(speech.tobytes())
    print(f"Speech Output Max: {np.max(np.frombuffer(out_speech, dtype=np.int16))} (Should be > 0)")
    print("\nModule 2 is working correctly.")
