# Audio Buffer Manager
# Manages buffering and silence detection for real-time audio streams.
import numpy as np
import collections


class AudioBufferManager:
    def __init__(self, frame_size_bytes=1024, silence_threshold=0.01, silence_frames_limit=15):
        """
        Manages buffering of audio chunks and detects end of speech.

        :param frame_size_bytes: Size of each incoming chunk in bytes.
        :param silence_threshold: Normalized RMS threshold to detect silence (0.0–1.0).
        :param silence_frames_limit: Consecutive silent frames before marking speech as complete.
        """
        self.buffer = collections.deque()
        self.silence_threshold = silence_threshold
        self.silence_frames_limit = silence_frames_limit
        self.silent_frame_count = 0
        self.is_currently_speaking = False
        self.utterance_buffer = []

    def process_chunk(self, audio_bytes: bytes):
        """
        Processes an audio chunk and tracks speech activity.

        :returns: (utterance_complete: bool, audio_bytes: bytes | None)
                  When an utterance ends, returns True and the full audio buffer.
        """
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_np ** 2))
        normalized_rms = rms / 32767.0
        is_speech = normalized_rms > self.silence_threshold

        if is_speech:
            self.is_currently_speaking = True
            self.silent_frame_count = 0
            self.utterance_buffer.append(audio_bytes)
        elif self.is_currently_speaking:
            self.silent_frame_count += 1
            self.utterance_buffer.append(audio_bytes)
            if self.silent_frame_count > self.silence_frames_limit:
                self.is_currently_speaking = False
                complete_utterance = b''.join(self.utterance_buffer)
                self.utterance_buffer = []
                return True, complete_utterance

        return False, None

    def get_current_buffer(self) -> bytes:
        """Returns the raw deque buffer as bytes."""
        return b''.join(self.buffer)

    def clear(self):
        """Resets all internal state."""
        self.buffer.clear()
        self.utterance_buffer = []
        self.silent_frame_count = 0
        self.is_currently_speaking = False


# --- Unit Test ---
if __name__ == "__main__":
    buffer_mgr = AudioBufferManager()
    silent_chunk = np.zeros(512, dtype=np.int16).tobytes()
    loud_chunk = (np.random.rand(512) * 32767).astype(np.int16).tobytes()
    print("Processing simulated stream...")
    for _ in range(3):
        buffer_mgr.process_chunk(silent_chunk)
    for _ in range(5):
        buffer_mgr.process_chunk(loud_chunk)
    for i in range(20):
        is_utt, data = buffer_mgr.process_chunk(silent_chunk)
        if is_utt:
            print(f"Utterance Complete! Size: {len(data)} bytes")
            break
