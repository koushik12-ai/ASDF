import numpy as np
import collections

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, aggressiveness=3, frame_duration_ms=30):
        """
        Initializes a Energy-Based VAD. (WebRTC replacement for Python 3.14 compatibility)
        
        :param aggressiveness: 0-3 scale. 
                               0 = Low threshold (very sensitive).
                               3 = High threshold (only loud speech).
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        
        # Calculate frame size in bytes (2 bytes per sample for Int16)
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        
        # Map aggressiveness to RMS thresholds
        # Higher aggressiveness = Higher threshold (harder to trigger)
        thresholds = {0: 0.01, 1: 0.02, 2: 0.04, 3: 0.08}
        self.threshold = thresholds.get(aggressiveness, 0.03)
        
        # --- State Machine Variables ---
        self.buffer = []
        self.triggered = False
        
        # Padding logic to avoid cutting words
        self.padding_frames = 10 
        self.ring_buffer = collections.deque(maxlen=self.padding_frames)
        self.num_unvoiced = 0

    def _is_speech(self, frame_bytes):
        """Determines if a frame contains speech based on energy (RMS)."""
        # Convert bytes to numpy array
        audio_np = np.frombuffer(frame_bytes, dtype=np.int16)
        
        # Calculate RMS (Root Mean Square) - Volume
        # Normalize to -1.0 to 1.0 range
        rms = np.sqrt(np.mean((audio_np.astype(np.float32) / 32768.0)**2))
        
        return rms > self.threshold

    def process_frame(self, audio_chunk_bytes):
        """
        Process a chunk of audio.
        Iterates through the chunk in valid frame sizes.
        
        Returns: 
            (is_speech_segment, audio_bytes)
        """
        offset = 0
        while offset + self.frame_size <= len(audio_chunk_bytes):
            frame = audio_chunk_bytes[offset : offset + self.frame_size]
            offset += self.frame_size
            
            result = self._process_single_frame(frame)
            if result is not None:
                return True, result
        
        return False, None

    def _process_single_frame(self, frame_bytes):
        """
        State machine logic.
        """
        is_speech = self._is_speech(frame_bytes)

        # State: NOT recording
        if not self.triggered:
            self.ring_buffer.append((frame_bytes, is_speech))
            num_voiced = len([f for f, speech in self.ring_buffer if speech])
            
            # If we detect enough speech frames, START recording
            if num_voiced > 0.6 * self.ring_buffer.maxlen:
                self.triggered = True
                self.buffer.extend([f for f, s in self.ring_buffer])
                self.ring_buffer.clear()
                return None
            
        # State: IS recording
        else:
            self.buffer.append(frame_bytes)
            
            if is_speech:
                self.num_unvoiced = 0
            else:
                self.num_unvoiced += 1
                
            # If silence exceeds padding, STOP recording
            if self.num_unvoiced >= self.padding_frames:
                self.triggered = False
                self.num_unvoiced = 0
                
                full_audio = b''.join(self.buffer)
                self.buffer = []
                return full_audio

        return None

    def get_current_buffer(self):
        if self.buffer:
            return b''.join(self.buffer)
        return None

# --- Unit Test ---
if __name__ == "__main__":
    print("Initializing Pure Python VAD...")
    vad = VoiceActivityDetector(aggressiveness=2)
    
    # Simulate Silence (Low numbers)
    silent_frame = (np.random.rand(int(vad.frame_size/2)) * 100).astype(np.int16).tobytes()
    
    # Simulate Speech (High numbers / Noise)
    speech_frame = (np.random.rand(int(vad.frame_size/2)) * 10000).astype(np.int16).tobytes()
    
    print("Feeding silence...")
    for _ in range(5):
        vad.process_frame(silent_frame)
        
    print("Feeding speech...")
    for _ in range(10):
        vad.process_frame(speech_frame)
        
    print("Feeding silence (padding)...")
    for i in range(12):
        is_segment, data = vad.process_frame(silent_frame)
        if is_segment:
            print(f"SUCCESS: VAD detected end of speech!")
            print(f"Segment size: {len(data)} bytes")
            break