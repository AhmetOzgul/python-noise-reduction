import numpy as np
import pyaudio
import time
from collections import deque
from fft import CustomFFT
from audio_recorder import AudioRecorder

class NoiseReducer:
    def __init__(self):
        self.RATE = 44100
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        
        self.recorder = AudioRecorder(self.RATE, self.CHANNELS)
        
        self.noise_samples = []
        self.noise_spectrum = None
        self.calibration_time = 0.1
        
        self.overlap_size = self.CHUNK // 2
        self.input_buffer = np.zeros(self.CHUNK + self.overlap_size)
        self.output_buffer = np.zeros(self.overlap_size)
        self.window = np.hanning(self.CHUNK)
        
        self.over_subtraction = 3.5
        self.min_gain = 0.01
        self.spectral_floor = 0.005
        self.multi_pass = True
        self.second_pass_factor = 1.5
        
        freqs = np.fft.fftfreq(self.CHUNK, 1/self.RATE)
        freqs = np.abs(freqs[:self.CHUNK//2+1])
        self.freq_weights = np.ones_like(freqs)
        self.freq_weights[freqs < 500] = 2.0
        self.freq_weights[(freqs >= 500) & (freqs < 2000)] = 1.5
        self.freq_weights[freqs >= 4000] = 1.8
        
        self.prev_gain = None
        self.gain_smoothing = 0.85
        
        self.output_queue = deque(maxlen=8)
        self.p = pyaudio.PyAudio()

    def create_noise_profile(self, data):
        if self.noise_spectrum is None:
            self.noise_samples.append(data.copy())
            total_samples = sum(len(x) for x in self.noise_samples)
            
            if (total_samples / self.RATE) >= self.calibration_time:
                noise_data = np.concatenate(self.noise_samples)
                n_frames = len(noise_data) // self.CHUNK
                spectrums = []
                
                for i in range(n_frames):
                    frame = noise_data[i * self.CHUNK:(i + 1) * self.CHUNK]
                    if len(frame) == self.CHUNK:
                        windowed = frame * self.window
                        spectrum = np.abs(CustomFFT.fft(windowed))
                        spectrums.append(spectrum)
                
                if spectrums:
                    self.noise_spectrum = np.mean(spectrums, axis=0)
                    print("Gürültü profili hazır")
                
                self.noise_samples = []

    def reduce_noise(self, data):
        if self.noise_spectrum is None:
            return data
        
        self.input_buffer[:-len(data)] = self.input_buffer[len(data):]
        self.input_buffer[-len(data):] = data
        
        windowed_signal = self.input_buffer[:self.CHUNK] * self.window
        
        spectrum = CustomFFT.fft(windowed_signal)
        
        half_spectrum = spectrum[:self.CHUNK//2+1]
        half_magnitude = np.abs(half_spectrum)
        half_phase = np.angle(half_spectrum)
        noise_estimate = self.noise_spectrum[:self.CHUNK//2+1]
        
        enhanced_noise = noise_estimate * self.over_subtraction * self.freq_weights
        
        signal_power = half_magnitude ** 2
        noise_power = enhanced_noise ** 2
        snr_ratio = signal_power / (noise_power + 1e-10)
        
        alpha = snr_ratio / (snr_ratio + 2)
        alpha = np.maximum(alpha, self.min_gain)
        
        if self.multi_pass:
            temp_magnitude = half_magnitude * alpha
            temp_snr = temp_magnitude / (noise_estimate + 1e-10)
            beta = temp_snr / (temp_snr + self.second_pass_factor)
            beta = np.maximum(beta, self.min_gain)
            final_gain = alpha * beta
        else:
            final_gain = alpha
        
        final_gain = np.maximum(final_gain, self.spectral_floor)
        
        if self.prev_gain is not None:
            final_gain = self.gain_smoothing * self.prev_gain + (1 - self.gain_smoothing) * final_gain
        self.prev_gain = final_gain.copy()
        
        clean_half_spectrum = half_magnitude * final_gain * np.exp(1j * half_phase)
        
        clean_spectrum = np.zeros_like(spectrum, dtype=complex)
        clean_spectrum[:self.CHUNK//2+1] = clean_half_spectrum
        clean_spectrum[self.CHUNK//2+1:] = np.conj(clean_half_spectrum[1:-1][::-1])
        
        clean_signal = np.real(CustomFFT.inverse_fft(clean_spectrum))
        
        output = clean_signal[:self.overlap_size] + self.output_buffer
        self.output_buffer = clean_signal[self.overlap_size:]
        
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output * min(0.9, 0.6 / max_val)
        
        output = np.tanh(output * 2) / 2
        
        return output.astype(np.float32)

    def input_callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.float32)
        
        self.recorder.add_raw_data(data)
        self.create_noise_profile(data)
        clean_data = self.reduce_noise(data)
        self.recorder.add_clean_data(clean_data)
        self.output_queue.append(clean_data)
        
        return (in_data, pyaudio.paContinue)

    def output_callback(self, in_data, frame_count, time_info, status):
        if self.output_queue:
            return (self.output_queue.popleft().tobytes(), pyaudio.paContinue)
        return (np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue)

    def start(self):
        self.stream_in = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK // 2,
            stream_callback=self.input_callback
        )
        
        self.stream_out = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            output=True,
            frames_per_buffer=self.CHUNK // 2,
            stream_callback=self.output_callback
        )
        
        print("Sistem başlatıldı.")
        print("Gürültü profili oluşturuluyor...")

    def stop(self):
        if hasattr(self, 'stream_in') and self.stream_in:
            self.stream_in.stop_stream()
            self.stream_in.close()
        if hasattr(self, 'stream_out') and self.stream_out:
            self.stream_out.stop_stream()
            self.stream_out.close()
        self.p.terminate()
        print("Sistem durduruldu.")

def main():
    reducer = NoiseReducer()
    try:
        reducer.start()
        print("\nKontroller:")
        print("R: Kayıt Başlat")
        print("S: Kayıt Durdur") 
        print("Q: Çıkış")
        
        while True:
            cmd = input().strip().upper()
            if cmd == 'R':
                reducer.recorder.start_recording()
            elif cmd == 'S':
                reducer.recorder.stop_recording()
            elif cmd == 'Q':
                break
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nKapatılıyor...")
    finally:
        if reducer.recorder.is_recording():
            reducer.recorder.stop_recording()
        reducer.stop()

if __name__ == "__main__":
    main() 