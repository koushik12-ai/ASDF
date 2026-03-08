import numpy as np

class NoiseSuppressor:
    def __init__(self, rate=16000, threshold_db=-40):
        """
        Initializes a Simple Noise Gate using only NumPy.
        
        :param rate: Sample rate.
        :param threshold_db: Volume threshold in dB. 
                             Sounds below this volume are considered noise and silenced.
                             -40dB is good for quiet rooms. 
                             -30dB is better for noisier rooms.
        """
        self.rate = rate
        # Convert dB threshold to amplitude (0.0 to 1.0)
        self.threshold = 10.0 ** (threshold_db / 20.0)

    def process(self, audio_chunk_bytes):
        """
        Applies noise gating: 
        If the signal is quiet (below threshold), mute it.
        If the signal is loud (above threshold), keep it.
        """
        # 1. Convert bytes to Int16 Array
        audio_np = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
        
        # 2. Normalize to -1.0 to 1.0 range
        audio_float = audio_np.astype(np.float32) / 32768.0
        
        # 3. Calculate the amplitude of the current chunk
        current_amplitude = np.max(np.abs(audio_float))
        
        # 4. Apply Noise Gate
        if current_amplitude < self.threshold:
            # It's noise (too quiet) -> Silence it
            processed_audio = np.zeros_like(audio_float)
        else:
            # It's speech -> Keep it (or optionally apply light smoothing)
            processed_audio = audio_float
            
        # 5. Convert back to Int16 bytes
        # Scale back up and clip to prevent distortion
        processed_audio_int = (processed_audio * 32768.0).astype(np.int16)
        
        return processed_audio_int.tobytes()

# --- Unit Test ---
if __name__ == "__main__":
    print("Testing NumPy Noise Suppressor...")
    
    # Create dummy signal
    # 1. Silence (Noise)
    silence = np.zeros(1000, dtype=np.int16)
    # 2. Speech (Loud noise)
    speech = np.random.randint(-10000, 10000, size=1000, dtype=np.int16)
    
    suppressor = NoiseSuppressor(threshold_db=-35) # Adjust threshold if needed
    
    # Test Silence
    out_silence = suppressor.process(silence.tobytes())
    print(f"Silence Input Max: {np.max(np.frombuffer(silence.tobytes(), dtype=np.int16))}")
    print(f"Silence Output Max: {np.max(np.frombuffer(out_silence, dtype=np.int16))} (Should be 0)")
    
    # Test Speech
    out_speech = suppressor.process(speech.tobytes())
    print(f"Speech Input Max: {np.max(np.frombuffer(speech.tobytes(), dtype=np.int16))}")
    print(f"Speech Output Max: {np.max(np.frombuffer(out_speech, dtype=np.int16))} (Should be > 0)")
    
    print("\nModule 2 is working correctly.")