import sounddevice as sd
import numpy as np

class AudioStreamingClient:
    def __init__(self, rate=16000, frames_per_buffer=512, channels=1):
        """
        Initialize the Audio Stream using SoundDevice.
        """
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.channels = channels
        self.dtype = 'int16'
        self.stream = None

    def start_stream(self):
        """Opens the microphone stream."""
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
            # Common error on Windows: missing host API
            if "Error opening InputStream" in str(e):
                 print("Please ensure your microphone is plugged in and drivers are installed.")

    def read_frame(self):
        """
        Reads a single frame of audio.
        Returns: bytes object containing audio data.
        """
        if self.stream and self.stream.active:
            # sounddevice returns a numpy array directly
            data_array, overflowed = self.stream.read(self.frames_per_buffer)
            
            if overflowed:
                print("[AudioClient] Audio overflow detected")
                
            # Convert numpy array back to bytes to match the pipeline requirement
            return data_array.tobytes()
            
        return None

    def stop_stream(self):
        """Stops and closes the stream."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("[AudioClient] Stream stopped.")

    def get_chunk_ms(self):
        """Helper to calculate latency in milliseconds."""
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