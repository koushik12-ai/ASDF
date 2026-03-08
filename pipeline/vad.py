# Module 3: Voice Activity Detection (VAD)
# Energy-based VAD that detects and isolates complete speech segments.
import numpy as np
import collections


class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, aggressiveness=3, frame_duration_ms=30):
        """
        Initializes an energy-based VAD (WebRTC replacement for Python 3.14+).

        :param sample_rate: Audio sample rate in Hz.
        :param aggressiveness: 0–3 scale.
                               0 = very sensitive (low threshold).
                               3 = only loud/clear speech (high threshold).
        :param frame_duration_ms: Duration of each analyzed frame in ms.
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms

        # Frame size in bytes (2 bytes per sample for Int16)
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)

        # Map aggressiveness level to RMS threshold
        thresholds = {0: 0.01, 1: 0.02, 2: 0.04, 3: 0.08}
        self.threshold = thresholds.get(aggressiveness, 0.03)

        # State machine variables
        self.buffer = []
        self.triggered = False
        self.padding_frames = 10
        self.ring_buffer = collections.deque(maxlen=self.padding_frames)
        self.num_unvoiced = 0

    def _is_speech(self, frame_bytes):
        """Determines if a frame contains speech based on RMS energy."""
        audio_np = np.frombuffer(frame_bytes, dtype=np.int16)
        rms = np.sqrt(np.mean((audio_np.astype(np.float32) / 32768.0) ** 2))
        return rms > self.threshold

    def process_frame(self, audio_chunk_bytes):
        """
        Process an audio chunk, iterating through valid frame sizes.

        :returns: (is_speech_segment: bool, audio_bytes: bytes | None)
        """
        offset = 0
        while offset + self.frame_size <= len(audio_chunk_bytes):
            frame = audio_chunk_bytes[offset: offset + self.frame_size]
            offset += self.frame_size
            result = self._process_single_frame(frame)
            if result is not None:
                return True, result
        return False, None

    def _process_single_frame(self, frame_bytes):
        """State machine: accumulates frames and emits a segment when speech ends."""
        is_speech = self._is_speech(frame_bytes)

        if not self.triggered:
            self.ring_buffer.append((frame_bytes, is_speech))
            num_voiced = len([f for f, speech in self.ring_buffer if speech])
            if num_voiced > 0.6 * self.ring_buffer.maxlen:
                self.triggered = True
                self.buffer.extend([f for f, s in self.ring_buffer])
                self.ring_buffer.clear()
        else:
            self.buffer.append(frame_bytes)
            if is_speech:
                self.num_unvoiced = 0
            else:
                self.num_unvoiced += 1
            if self.num_unvoiced >= self.padding_frames:
                self.triggered = False
                self.num_unvoiced = 0
                full_audio = b''.join(self.buffer)
                self.buffer = []
                return full_audio

        return None

    def get_current_buffer(self):
        """Returns the current accumulated audio buffer as bytes."""
        if self.buffer:
            return b''.join(self.buffer)
        return None


# --- Unit Test ---
if __name__ == "__main__":
    print("Initializing Pure Python VAD...")
    vad = VoiceActivityDetector(aggressiveness=2)
    silent_frame = (np.random.rand(int(vad.frame_size / 2)) * 100).astype(np.int16).tobytes()
    speech_frame = (np.random.rand(int(vad.frame_size / 2)) * 10000).astype(np.int16).tobytes()
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
            print(f"SUCCESS: VAD detected end of speech! Segment size: {len(data)} bytes")
            break
