import sounddevice as sd
import numpy as np

class AudioStreamingClient:
    def __init__(self, rate=16000, frames_per_buffer=512, channels=1):
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.channels = channels
        self.dtype = 'int16'
        self.stream = None

    def start_stream(self):
        try:
            self.stream = sd.InputStream(samplerate=self.rate, blocksize=self.frames_per_buffer, dtype=self.dtype, channels=self.channels)
            self.stream.start()
            print(f"[AudioClient] Stream started: {self.rate}Hz")
        except Exception as e:
            print(f"[AudioClient] Error: {e}")

    def read_frame(self):
        if self.stream and self.stream.active:
            data_array, overflowed = self.stream.read(self.frames_per_buffer)
            return data_array.tobytes()
        return None

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()