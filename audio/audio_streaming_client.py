# Audio Streaming Client
# Standalone helper for opening and reading from a live microphone stream.
import sounddevice as sd
import numpy as np


class AudioStreamingClient:
    def __init__(self, rate=16000, frames_per_buffer=512, channels=1):
        """
        Initialize the audio stream using SoundDevice.

        :param rate: Sample rate in Hz (default 16000).
        :param frames_per_buffer: Number of frames per read call.
        :param channels: Number of audio channels (1 = mono).
        """
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.channels = channels
        self.dtype = 'int16'
        self.stream = None

    def start_stream(self):
        """Opens the microphone input stream."""
        try:
            self.stream = sd.InputStream(
                samplerate=self.rate,
                blocksize=self.frames_per_buffer,
                dtype=self.dtype,
                channels=self.channels
            )
            self.stream.start()
            print(f"[AudioClient] Stream started: {self.rate}Hz, Buffer: {self.frames_per_buffer} frames")
        except Exception as e:
            print(f"[AudioClient] Error opening stream: {e}")
            if "Error opening InputStream" in str(e):
                print("Please ensure your microphone is plugged in and drivers are installed.")

    def read_frame(self):
        """
        Reads a single frame of audio.
        :returns: bytes object with raw audio data, or None if stream is inactive.
        """
        if self.stream and self.stream.active:
            data_array, overflowed = self.stream.read(self.frames_per_buffer)
            if overflowed:
                print("[AudioClient] Audio overflow detected")
            return data_array.tobytes()
        return None

    def stop_stream(self):
        """Stops and closes the microphone stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("[AudioClient] Stream stopped.")

    def get_chunk_ms(self) -> float:
        """Returns the duration of each audio chunk in milliseconds."""
        return (self.frames_per_buffer / self.rate) * 1000


# --- Unit Test ---
if __name__ == "__main__":
    client = AudioStreamingClient()
    client.start_stream()
    print("Recording 5 chunks for testing...")
    for i in range(5):
        data = client.read_frame()
        print(f"Chunk {i+1}: Size {len(data)} bytes")
    client.stop_stream()
