import numpy as np
import collections

class AudioBufferManager:
    def __init__(self, frame_size_bytes=1024, silence_threshold=0.01, silence_frames_limit=15):
        """
        Manages the buffering of audio chunks.
        :param frame_size_bytes: Size of each incoming chunk in bytes.
        :param silence_threshold: RMS amplitude threshold to detect silence.
        :param silence_frames_limit: Number of consecutive silent frames to consider 'end of speech'.
        """
        self.buffer = collections.deque()
        self.silence_threshold = silence_threshold
        self.silence_frames_limit = silence_frames_limit
        
        # State variables
        self.silent_frame_count = 0
        self.is_currently_speaking = False
        self.utterance_buffer = []

    def process_chunk(self, audio_bytes):
        """
        Adds a chunk to the buffer and checks for speech activity.
        Returns: (is_speech_active, frame_data)
        """
        # Convert bytes to numpy array for processing
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Calculate RMS (Root Mean Square) to determine volume
        rms = np.sqrt(np.mean(audio_np**2))
        normalized_rms = rms / 32767.0 # Normalize to 0.0 - 1.0 range
        
        is_speech = normalized_rms > self.silence_threshold

        # Basic Voice Activity Logic
        if is_speech:
            self.is_currently_speaking = True
            self.silent_frame_count = 0
            self.utterance_buffer.append(audio_bytes)
        elif self.is_currently_speaking:
            # Speech was happening, now we see silence
            self.silent_frame_count += 1
            self.utterance_buffer.append(audio_bytes) # Keep silence briefly to avoid cutting words
            
            if self.silent_frame_count > self.silence_frames_limit:
                # User stopped speaking
                self.is_currently_speaking = False
                complete_utterance = b''.join(self.utterance_buffer)
                self.utterance_buffer = [] # Reset for next turn
                return True, complete_utterance # Signal that an utterance is ready

        return False, None

    def get_current_buffer(self):
        """Returns the raw buffer (useful for continuous streaming modes)."""
        return b''.join(self.buffer)

    def clear(self):
        """Clears the buffer."""
        self.buffer.clear()
        self.utterance_buffer = []
        self.silent_frame_count = 0
        self.is_currently_speaking = False

# --- Unit Test ---
if __name__ == "__main__":
    # Simulate audio input
    buffer_mgr = AudioBufferManager()
    
    # Simulate 10 chunks: 5 silent, 5 loud
    silent_chunk = np.zeros(512, dtype=np.int16).tobytes()
    loud_chunk = (np.random.rand(512) * 32767).astype(np.int16).tobytes()
    
    print("Processing simulated stream...")
    
    # Feed silent
    for _ in range(3):
        buffer_mgr.process_chunk(silent_chunk)
        
    # Feed loud (simulate speech)
    for _ in range(5):
        is_utt, data = buffer_mgr.process_chunk(loud_chunk)
        
    # Feed silent (simulate pause)
    for i in range(20):
        is_utt, data = buffer_mgr.process_chunk(silent_chunk)
        if is_utt:
            print(f"Utterance Complete! Size: {len(data)} bytes")
            break