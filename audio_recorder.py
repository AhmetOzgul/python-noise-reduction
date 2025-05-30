import wave
import numpy as np
import os
from datetime import datetime

class AudioRecorder:
    def __init__(self, rate=44100, channels=1):
        self.RATE = rate
        self.CHANNELS = channels
        self.recording = False
        self.raw_frames = []
        self.clean_frames = []
    
    def start_recording(self):
        self.recording = True
        self.raw_frames = []
        self.clean_frames = []
        self.start_time = datetime.now()
        print("Kayıt başladı...")
    
    def add_raw_data(self, data):
        if self.recording:
            self.raw_frames.extend(data)
    
    def add_clean_data(self, data):
        if self.recording:
            self.clean_frames.extend(data)
    
    def stop_recording(self):
        if not self.recording:
            return
            
        self.recording = False
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        os.makedirs("recordings", exist_ok=True)
        
        raw_file = f"recordings/raw_{timestamp}.wav"
        clean_file = f"recordings/clean_{timestamp}.wav"
        
        self._save_wav(raw_file, self.raw_frames)
        self._save_wav(clean_file, self.clean_frames)
        
        print(f"Kayıt tamamlandı! Ham: {raw_file} | Temiz: {clean_file}")
    
    def _save_wav(self, filename, data):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.RATE)
            audio_data = (np.array(data) * 32767).astype(np.int16)
            wf.writeframes(audio_data.tobytes())
    
    def is_recording(self):
        return self.recording 