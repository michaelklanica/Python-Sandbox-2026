import sys
import numpy as np
import sounddevice as sd
import soundfile as sf
import json
import threading
import os
from datetime import datetime
from scipy import signal
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QLabel, QPushButton, 
                             QFileDialog, QFrame, QGroupBox, QSpinBox, 
                             QComboBox, QCheckBox, QMessageBox, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QRect, QObject, pyqtSlot, QTimer
from PyQt6.QtGui import QPainter, QPen, QColor, QPolygonF, QPainterPath, QFont, QImage, QLinearGradient

# --- 1. DSP ENGINE: High-Fidelity Oscillators & Filters ---

class PolyBLEPOscillator:
    """
    Anti-Aliased Oscillator using Polynomial Band-Limited Step (PolyBLEP).
    Optimized for AVX2 via NumPy vectorization. Supports Stereo Arrays.
    """
    @staticmethod
    def _blep(t, dt):
        """
        PolyBLEP smoothing function.
        """
        # 0 < t < 1
        return np.where(t < dt, 
                        # Near 0: Smooth the jump
                        t**2 / dt - t - 0.5, 
                        np.where(t > 1 - dt, 
                                 # Near 1: Smooth the wrap-around
                                 (t - 1)**2 / dt + (t - 1) + 0.5, 
                                 # Elsewhere: No correction
                                 0.0))

    @staticmethod
    def generate(waveform, phase, inc):
        """
        Generates band-limited waveforms.
        phase: Normalized phase array [0.0, 1.0] (Can be N x 2 for stereo)
        inc: Phase increment array (freq / sr)
        """
        # Wrap phase to [0, 1) just in case
        phase = phase % 1.0
        
        if waveform == "Sine":
            return np.sin(2 * np.pi * phase)
            
        elif waveform == "Sawtooth":
            # Naive Saw: 2 * phase - 1 (Range -1 to 1)
            naive = 2.0 * phase - 1.0
            # Apply correction
            correction = PolyBLEPOscillator._blep(phase, inc)
            # Subtract BLEP * 2 (because step height is 2)
            return naive - 2.0 * correction
            
        elif waveform == "Square":
            # Naive Square: +1 if phase < 0.5 else -1
            naive = np.where(phase < 0.5, 1.0, -1.0)
            # Square is sum of two steps. One at 0.0, one at 0.5.
            # Correction at 0.0
            blep0 = PolyBLEPOscillator._blep(phase, inc)
            # Correction at 0.5 (shift phase by 0.5)
            phase_shifted = (phase + 0.5) % 1.0
            blep05 = PolyBLEPOscillator._blep(phase_shifted, inc)
            
            return naive + blep0 - blep05
            
        elif waveform == "Triangle":
            # High quality triangle is derived from integrated square
            return 4.0 * np.abs(phase - 0.5) - 1.0
            
        return np.zeros_like(phase)

class BiquadFilter:
    """
    Digital Biquad Filter supporting Lowpass, Highpass, and Bandpass.
    Uses RBJ Audio EQ Cookbook formulas.
    """
    @staticmethod
    def process(data, cutoff_arr, sample_rate, q=0.707, f_type="Lowpass"):
        # Limit cutoff to prevent explosion near Nyquist
        cutoff = np.clip(cutoff_arr, 20, sample_rate * 0.49)
        
        # Intermediate variables (Vectorized)
        w0 = 2 * np.pi * cutoff / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2 * q)
        
        # Use mean cutoff for static filtering
        mean_cut = np.mean(cutoff)
        w0_s = 2 * np.pi * mean_cut / sample_rate
        cos_s = np.cos(w0_s)
        sin_s = np.sin(w0_s)
        alpha_s = sin_s / (2 * q)
        a0 = 1 + alpha_s

        if f_type == "Lowpass":
            b0 = (1 - cos_s) / 2
            b1 = 1 - cos_s
            b2 = (1 - cos_s) / 2
            a1 = -2 * cos_s
            a2 = 1 - alpha_s
        elif f_type == "Highpass":
            b0 = (1 + cos_s) / 2
            b1 = -(1 + cos_s)
            b2 = (1 + cos_s) / 2
            a1 = -2 * cos_s
            a2 = 1 - alpha_s
        elif f_type == "Bandpass":
            b0 = alpha_s
            b1 = 0
            b2 = -alpha_s
            a1 = -2 * cos_s
            a2 = 1 - alpha_s
        else:
            return data

        # Normalize
        coeffs_b = np.array([b0, b1, b2]) / a0
        coeffs_a = np.array([1.0, a1/a0, a2/a0])
        
        # Apply along axis 0 (time), handling stereo channels independently
        return signal.lfilter(coeffs_b, coeffs_a, data, axis=0)

# --- 2. WORKER THREADS ---

class SynthesisWorker(QObject):
    result_ready = pyqtSignal(object, object, object, float) 

    def __init__(self, engine_config):
        super().__init__()
        self.config = engine_config
        self.sample_rate = 44100 

    @pyqtSlot(dict)
    def render(self, params):
        try:
            sample_rate = params.get('sample_rate', 44100)
            layers = params.get('layers', [])
            drive = params.get('drive', 1.0)
            
            # Duration Calculation
            max_total_ms = 50.0
            for layer in layers:
                if layer['active']:
                    amp_ms = layer['amp_attack'] + layer['amp_decay'] + 150 + layer['amp_release']
                    pitch_ms = 1.0 + (layer['pitch_env_dec'] * 3) + (layer['pitch_env_dec'] * 5)
                    max_total_ms = max(max_total_ms, amp_ms, pitch_ms)
            
            max_total_ms += 20.0
            duration = max_total_ms / 1000.0
            num_samples = int(sample_rate * duration)
            
            if num_samples <= 0:
                self.result_ready.emit(np.zeros((1, 2)), np.zeros(1), np.zeros(1), 0.0)
                return

            # Initialize Stereo Buffer (N x 2)
            total_sig = np.zeros((num_samples, 2))
            composite_pitch_env = np.zeros(num_samples)
            active_count = 0

            # Generate Layers
            for layer in layers:
                if not layer['active']: continue
                
                active_count += 1
                
                # --- Fast ADSR Generator ---
                def gen_adsr(length, a_ms, d_ms, s_lvl, r_ms):
                    sr = sample_rate
                    a_s = int((a_ms / 1000.0) * sr)
                    d_s = int((d_ms / 1000.0) * sr)
                    r_s = int((r_ms / 1000.0) * sr)
                    a_s, d_s, r_s = max(1, a_s), max(1, d_s), max(1, r_s)
                    env = np.zeros(length)
                    
                    # Attack
                    a_end = min(a_s, length)
                    env[:a_end] = np.linspace(0, 1, a_end)
                    
                    if a_end < length:
                        # Decay
                        d_end = min(a_end + d_s, length)
                        if d_end > a_end:
                            t_d = np.linspace(0, 1, d_end - a_end)
                            decay_curve = (np.exp(-5 * t_d) - 0.0067) / 0.9933
                            env[a_end:d_end] = s_lvl + (1.0 - s_lvl) * decay_curve
                        
                        # Sustain
                        r_start = max(d_end, length - r_s)
                        if r_start > d_end:
                            env[d_end:r_start] = s_lvl
                        
                        # Release
                        if r_start < length:
                            t_r = np.linspace(0, 1, length - r_start)
                            rel_curve = (np.exp(-5 * t_r) - 0.0067) / 0.9933
                            env[r_start:] = s_lvl * rel_curve
                    return env

                # --- Pitch Envelope (Mono calculation used for base) ---
                p_env = gen_adsr(num_samples, 1.0, layer['pitch_env_dec'] * 3, 0.0, layer['pitch_env_dec'] * 5)
                f_env = layer['base_pitch'] + layer['pitch_env_int'] * p_env
                composite_pitch_env += f_env
                
                # --- Stereo Spread Logic ---
                spread = layer.get('spread', 0.0)
                
                # Setup Frequency for L/R
                if spread > 0.01:
                    # Detune Right channel slightly up, Left slightly down (max approx 2%)
                    detune_factor = spread * 0.02 
                    f_L = f_env * (1.0 - detune_factor * 0.5)
                    f_R = f_env * (1.0 + detune_factor * 0.5)
                    freq_hz = np.stack([f_L, f_R], axis=1) # (N, 2)
                else:
                    # Mono frequency
                    freq_hz = np.stack([f_env, f_env], axis=1)

                # --- Frequency Modulation ---
                # Apply same mod index to both, but frequencies differ slightly if spread
                mod_phase = np.cumsum(freq_hz * layer['fm_ratio'] / sample_rate, axis=0)
                modulator = np.sin(2 * np.pi * mod_phase) * layer['fm_amount']
                
                carrier_freq = freq_hz + modulator
                carrier_freq = np.maximum(carrier_freq, 1.0)
                
                inc = carrier_freq / sample_rate
                phase = np.cumsum(inc, axis=0)
                
                # --- PolyBLEP Oscillator (Stereo Aware) ---
                osc = PolyBLEPOscillator.generate(layer['waveform_type'], phase, inc)

                # --- Noise Logic (Stereo Decorrelation) ---
                # Generate two independent noise buffers
                n1 = np.random.uniform(-1, 1, num_samples)
                if spread > 0.01:
                    n2 = np.random.uniform(-1, 1, num_samples)
                    # Mix N1 and N2 based on spread for Right channel
                    # Spread 0 = Perfectly Mono (n1, n1)
                    # Spread 1 = Perfectly Stereo (n1, n2)
                    noise_L = n1
                    noise_R = n1 * (1.0 - spread) + n2 * spread
                    
                    # Normalize R energy roughly
                    norm = np.max(np.abs(noise_R)) + 1e-9
                    if norm > 0: noise_R /= norm
                    
                    noise_raw = np.stack([noise_L, noise_R], axis=1)
                else:
                    noise_raw = np.stack([n1, n1], axis=1)

                # Apply Filter Color to Noise (Pink/Brown)
                if layer['noise_type'] == "Pink":
                    # Simple Pink approximation filter on the stereo buffer
                    b_pink = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                    a_pink = [1.0, -2.494956002, 2.017265875, -0.522189400]
                    noise_raw = signal.lfilter(b_pink, a_pink, noise_raw, axis=0)
                elif layer['noise_type'] == "Brown":
                    noise_raw = np.cumsum(noise_raw, axis=0)
                    # Remove DC drift
                    noise_raw -= np.mean(noise_raw, axis=0)
                
                # Normalize Noise
                nm = np.max(np.abs(noise_raw))
                if nm > 0: noise_raw /= nm

                # --- Amplitude Envelope ---
                a_env = gen_adsr(num_samples, layer['amp_attack'], layer['amp_decay'], layer['amp_sustain'], layer['amp_release'])
                # Reshape for broadcasting (N, 1) to multiply (N, 2)
                a_env = a_env[:, np.newaxis]

                # --- Filter (Osc + Noise) ---
                raw_sig = osc * (1.0 - layer['noise_mix']) + noise_raw * layer['noise_mix']
                
                # Use Global Filter
                q_val = layer.get('noise_q', 1.0)
                f_type = layer.get('filter_type', "Lowpass")
                # Filter processes (N, 2) automatically along axis 0
                filtered_sig = BiquadFilter.process(raw_sig, layer['noise_cutoff'], sample_rate, q=q_val, f_type=f_type)
                
                # --- Bitcrusher ---
                crush_amt = layer.get('bit_crush', 0.0)
                if crush_amt > 0.0:
                    step = int(1 + crush_amt * 20)
                    if step > 1:
                        indices = np.arange(0, len(filtered_sig), step)
                        if len(indices) > 0:
                            reduced = filtered_sig[indices]
                            # Repeat elements (works on both channels)
                            filtered_sig = np.repeat(reduced, step, axis=0)[:len(filtered_sig)]

                # Apply Amp Envelope
                layer_sig = filtered_sig * a_env
                
                total_sig += layer_sig * layer['gain']

            # --- Post Processing ---
            if active_count > 0:
                pitch_data = (composite_pitch_env / active_count).astype(np.float32)
            else:
                pitch_data = np.zeros(num_samples).astype(np.float32)

            # Apply Saturation
            sig = np.tanh(total_sig * drive)
            
            # Makeup Gain
            sig *= 1.3

            # Master Fade
            fade_len = int(sample_rate * 0.010) 
            if len(sig) > fade_len:
                fade_curve = np.linspace(1.0, 0.0, fade_len)[:, np.newaxis]
                sig[-fade_len:] *= fade_curve
                sig[-1] = 0.0
            
            # Spectrum Analysis (Mix to Mono for display)
            sig_mono = np.mean(sig, axis=1)
            n_fft = 2048
            window = np.hanning(min(len(sig_mono), n_fft))
            fft_res = np.abs(np.fft.rfft(sig_mono[:len(window)] * window))
            spec_data = 20 * np.log10(fft_res + 1e-7)

            self.result_ready.emit(sig.astype(np.float32), spec_data, pitch_data, duration)

        except Exception as e:
            print(f"Synthesis Error: {e}")

class SpectrogramWorker(QThread):
    finished = pyqtSignal(object)
    def __init__(self, audio_data, sample_rate, color_table):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.color_table = color_table

    def run(self):
        # Mix to mono for visualization if stereo
        if self.audio_data.ndim > 1 and self.audio_data.shape[1] == 2:
            mono_data = np.mean(self.audio_data, axis=1)
        else:
            mono_data = self.audio_data

        if mono_data.size <= 1:
            self.finished.emit(None); return
            
        nperseg = min(len(mono_data), 256)
        f, t_spec, Sxx = signal.spectrogram(mono_data, self.sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
        data = 10 * np.log10(Sxx + 1e-10)
        norm_data = np.clip((data - -100) / (-20 - -100) * 255, 0, 255).astype(np.uint8)
        norm_data = np.flipud(norm_data).copy()
        img_h, img_w = norm_data.shape
        img = QImage(norm_data.data, img_w, img_h, img_w, QImage.Format.Format_Indexed8)
        img.setColorTable(self.color_table)
        self.finished.emit(img.copy())

# --- 3. HARDWARE & UI INTEGRATION ---

class AudioEngine:
    def __init__(self):
        self.master_vol = 1.0
        self.active_voices = []
        self.cached_sample = np.zeros((1, 2)) # Stereo init
        self.lock = threading.Lock()
        self.setup_audio_device()
        self.color_table = [QColor.fromHsl(int(240 - (i/255.0 * 180)), 200, min(150, i//2+20)).rgb() for i in range(256)]

    def setup_audio_device(self):
        try:
            apis = sd.query_hostapis()
            jack_index = -1
            for i, api in enumerate(apis):
                if 'JACK' in api['name']:
                    jack_index = i
                    break
            
            device = None
            if jack_index >= 0:
                print(f"Found JACK/PipeWire API at index {jack_index}.")
                host_devices = sd.query_devices()
                for i, d in enumerate(host_devices):
                    if d['hostapi'] == jack_index and d['max_output_channels'] > 0:
                        device = i
                        break
            
            if device is not None:
                dev_info = sd.query_devices(device)
                self.sample_rate = int(dev_info['default_samplerate'])
                print(f"Using Audio Device: {dev_info['name']} @ {self.sample_rate}Hz")
                # Channels=2 for Stereo
                self.stream = sd.OutputStream(samplerate=self.sample_rate, device=device, channels=2, callback=self.audio_callback, blocksize=512, latency='low')
            else:
                dev_info = sd.query_devices(kind='output')
                self.sample_rate = int(dev_info['default_samplerate'])
                print(f"Fallback Device: {dev_info['name']} @ {self.sample_rate}Hz")
                self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=2, callback=self.audio_callback, blocksize=512)
                
        except Exception as e:
            print(f"Audio Init Failed: {e}")
            self.sample_rate = 44100

    def trigger(self):
        with self.lock:
            if self.cached_sample.size > 0:
                if len(self.active_voices) > 16:
                    self.active_voices.pop(0) 
                self.active_voices.append([self.cached_sample, 0])

    def audio_callback(self, outdata, frames, time, status):
        # Out buffer is now (frames, 2)
        out = np.zeros((frames, 2))
        with self.lock:
            keep = []
            for voice in self.active_voices:
                buf, ptr = voice
                rem = len(buf) - ptr
                take = min(frames, rem)
                
                # Add stereo voice to stereo out
                out[:take] += buf[ptr : ptr + take]
                
                voice[1] += take
                if voice[1] < len(buf):
                    keep.append(voice)
            self.active_voices = keep
        outdata[:] = np.clip(out * self.master_vol, -1, 1)

class StaticVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(420)
        # Vis buffers remain mono 1D for simplicity
        self.waveform = np.zeros(1); self.spectrum = np.zeros(1); self.pitch_env = np.zeros(1)
        self.spectrogram_img = None
        self.duration = 0.0
        self.show_heat = True; self.show_wave = True; self.show_pitch = True; self.show_harmonics = True

    def update_layers(self, wave, spec, pitch, dur):
        # If wave is stereo, mix to mono for display
        if wave.ndim > 1 and wave.shape[1] == 2:
            self.waveform = np.mean(wave, axis=1)
        else:
            self.waveform = wave
            
        self.spectrum = spec; self.pitch_env = pitch; self.duration = dur
        self.update()

    def set_spectrogram(self, img):
        self.spectrogram_img = img; self.update()

    def paintEvent(self, event):
        qp = QPainter(self); qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QLinearGradient(0, 0, 0, self.height())
        bg.setColorAt(0, QColor(15, 15, 20)); bg.setColorAt(1, QColor(5, 5, 7))
        qp.fillRect(self.rect(), bg)

        w, h = self.width(), self.height()
        m_l, m_r, m_t, m_b = 80, 80, 40, 60
        draw_w, draw_h = w - m_l - m_r, h - m_t - m_b
        mid_y = m_t + (draw_h // 2)

        if self.show_heat and self.spectrogram_img:
            qp.setOpacity(0.4)
            qp.drawImage(QRect(m_l, m_t, draw_w, draw_h), self.spectrogram_img)
            qp.setOpacity(1.0)

        if self.show_harmonics and self.spectrum.size > 1:
            h_path = QPainterPath(); h_path.moveTo(m_l, h - m_b)
            l_min, l_max = np.log10(20), np.log10(20000)
            spec_max = np.max(self.spectrum) if self.spectrum.size > 0 else 1.0
            v_max_db = max(0.0, spec_max + 6.0)
            step = max(1, len(self.spectrum) // draw_w) 
            for i in range(1, len(self.spectrum), step):
                freq = i * (44100 / 2 / len(self.spectrum))
                if freq < 20: continue
                x = m_l + ((np.log10(freq) - l_min) / (l_max - l_min)) * draw_w
                val = np.interp(self.spectrum[i], [-100, v_max_db], [0, 1])
                y = (h - m_b) - (val * draw_h)
                h_path.lineTo(x, y)
            h_path.lineTo(m_l + draw_w, h - m_b)
            qp.fillPath(h_path, QColor(249, 115, 22, 25))
            qp.setPen(QPen(QColor(249, 115, 22, 150), 1))
            qp.drawPath(h_path)

        if self.show_wave and self.waveform.size > 1:
            qp.setPen(QPen(QColor(253, 224, 71, 180), 1.2))
            w_poly = QPolygonF()
            step = max(1, len(self.waveform) // draw_w)
            for i in range(0, len(self.waveform), step):
                x_pos = m_l + (i / len(self.waveform)) * draw_w
                y = mid_y - (self.waveform[i] * (draw_h // 2))
                w_poly.append(QPointF(x_pos, y))
            qp.drawPolyline(w_poly)

        if self.show_pitch and self.pitch_env.size > 1:
            qp.setPen(QPen(QColor(239, 68, 68), 2))
            p_poly = QPolygonF()
            p_min, p_max = np.min(self.pitch_env), np.max(self.pitch_env)
            l_p_min, l_p_max = np.log10(max(1, p_min*0.9)), np.log10(max(10, p_max*1.1))
            step = max(1, len(self.pitch_env) // draw_w)
            for i in range(0, len(self.pitch_env), step):
                x_pos = m_l + (i / len(self.pitch_env)) * draw_w
                val = np.interp(np.log10(max(1, self.pitch_env[i])), [l_p_min, l_p_max], [0, 1])
                p_poly.append(QPointF(x_pos, (h - m_b) - (val * draw_h)))
            qp.drawPolyline(p_poly)
            
        qp.setPen(QPen(QColor(255, 255, 255, 20), 1))
        qp.drawRect(m_l, m_t, draw_w, draw_h)
        qp.drawLine(m_l, mid_y, m_l + draw_w, mid_y)
        qp.setFont(QFont("Segoe UI Semibold", 8)); qp.setPen(QColor(120, 120, 130))
        qp.drawText(m_l, h - 35, "0ms")
        qp.drawText(m_l + draw_w - 50, h - 35, f"{int(self.duration*1000)}ms")

class DrumSynthApp(QMainWindow):
    synth_params_changed = pyqtSignal(dict) 

    def __init__(self):
        super().__init__()
        self.engine = AudioEngine()
        
        self.synth_thread = QThread()
        self.synth_worker = SynthesisWorker({})
        self.synth_worker.moveToThread(self.synth_thread)
        self.synth_params_changed.connect(self.synth_worker.render)
        self.synth_worker.result_ready.connect(self.handle_synthesis_result)
        self.synth_thread.start()

        self.spec_worker = None
        self.layers = [self.get_default_layer_params(i) for i in range(3)]
        self.drive_val = 1.0
        
        self.presets_dir = os.path.join(os.getcwd(), "drum-presets")
        self.samples_dir = os.path.join(os.getcwd(), "drum-samples")
        os.makedirs(self.presets_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        
        # --- Debounce Timer to prevent UI Freeze ---
        self.recalc_timer = QTimer()
        self.recalc_timer.setInterval(50) # 50ms delay
        self.recalc_timer.setSingleShot(True)
        self.recalc_timer.timeout.connect(self.trigger_recalc)

        self.ui_ready = False
        self.init_ui()
        self.engine.stream.start()
        self.ui_ready = True
        self.update_ui_from_layer()
        self.trigger_recalc() 

    def get_default_layer_params(self, index):
        # LAYER 1: The Body (Tonal Punch)
        if index == 0:
            return {
                "active": True,
                "gain": 0.9, 
                "waveform_type": "Sine", 
                "base_pitch": 185.0, # Snare fundamental
                "pitch_env_int": 50.0, 
                "pitch_env_dec": 60.0, 
                "fm_ratio": 1.0, 
                "fm_amount": 0.0,
                "amp_attack": 0.0, 
                "amp_decay": 180.0, 
                "amp_sustain": 0.0, 
                "amp_release": 20.0, 
                "noise_type": "White", 
                "noise_mix": 0.0, 
                "noise_cutoff": 1000.0, 
                "noise_q": 1.0,
                "filter_type": "Lowpass", 
                "bit_crush": 0.0,
                "spread": 0.0 
            }
        
        # LAYER 2: The Wires (Filtered Noise)
        elif index == 1:
            return {
                "active": True,
                "gain": 0.65, 
                "waveform_type": "Sine", 
                "base_pitch": 100.0, 
                "pitch_env_int": 0.0, 
                "pitch_env_dec": 0.0, 
                "fm_ratio": 1.0, 
                "fm_amount": 0.0,
                "amp_attack": 0.0, 
                "amp_decay": 300.0, # Longer tail
                "amp_sustain": 0.0, 
                "amp_release": 100.0, 
                "noise_type": "Pink", 
                "noise_mix": 1.0, # Pure noise
                "noise_cutoff": 3500.0, # Mid-high sizzle
                "noise_q": 0.8,
                "filter_type": "Bandpass", 
                "bit_crush": 0.0,
                "spread": 0.6 # Wide spread for noise
            }

        # LAYER 3: The Transient (Crack/Click)
        else:
            return {
                "active": True,
                "gain": 0.5, 
                "waveform_type": "Square", 
                "base_pitch": 800.0, 
                "pitch_env_int": 500.0, # Sharp drop
                "pitch_env_dec": 15.0, 
                "fm_ratio": 3.5, 
                "fm_amount": 150.0, # Metallic inharmonicity
                "amp_attack": 0.0, 
                "amp_decay": 40.0, # Very short
                "amp_sustain": 0.0, 
                "amp_release": 10.0, 
                "noise_type": "White", 
                "noise_mix": 0.2, 
                "noise_cutoff": 5000.0, 
                "noise_q": 1.0,
                "filter_type": "Highpass", 
                "bit_crush": 0.1,
                "spread": 0.0
            }

    @pyqtSlot(object, object, object, float)
    def handle_synthesis_result(self, sample, spec, pitch, duration):
        with self.engine.lock:
            self.engine.cached_sample = sample
        self.vis.update_layers(sample, spec, pitch, duration)
        
        if self.spec_worker and self.spec_worker.isRunning():
            self.spec_worker.terminate()
        self.spec_worker = SpectrogramWorker(sample, self.engine.sample_rate, self.engine.color_table)
        self.spec_worker.finished.connect(self.vis.set_spectrogram)
        self.spec_worker.start()
        
        main_f = self.layers[0]["base_pitch"] if self.layers[0]["active"] else 0
        self.note_lbl.setText(self.get_note_name(main_f))

    def trigger_recalc(self):
        if not self.ui_ready: return
        params = {
            'sample_rate': self.engine.sample_rate,
            'drive': self.drive_val,
            'layers': [l.copy() for l in self.layers] 
        }
        self.synth_params_changed.emit(params)

    def set_master_vol(self, v):
        self.engine.master_vol = v / 100.0

    def init_ui(self):
        self.setWindowTitle("Drum Architect Pro [Stereo Engine]"); self.setMinimumWidth(1150)
        self.setStyleSheet("""
            QMainWindow { background-color: #08080a; color: #e2e2e2; } 
            QFrame#panel { background: #111115; border: 1px solid #1f1f25; border-radius: 6px; } 
            QLabel { color: #71717a; font-weight: 600; }
            QPushButton#tgl { background: transparent; color: #52525b; border: 1px solid #27272a; padding: 4px; font-size: 10px; font-weight: bold; }
            QPushButton#tgl:checked { background: #27272a; color: #f97316; border-color: #f97316; }
            QSpinBox { background: #18181b; color: #f97316; border: 1px solid #27272a; padding: 2px; }
            QTabWidget::pane { border: 1px solid #1f1f25; background: #111115; }
            QTabBar::tab { background: #18181b; color: #71717a; padding: 8px 20px; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; }
            QTabBar::tab:selected { background: #27272a; color: #f97316; font-weight: bold; }
            QGroupBox { border: 1px solid #1f1f25; margin-top: 1.2em; border-radius: 4px; padding-top: 10px; font-weight: bold; color: #666; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
        """)
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central); layout.setContentsMargins(25, 25, 25, 25)

        vis_frame = QFrame(); vis_frame.setObjectName("panel")
        vis_lay = QVBoxLayout(vis_frame)
        
        # --- Visualization Toggles ---
        vis_tools = QHBoxLayout(); vis_tools.setSpacing(6)
        self.btn_spectro = self.create_vis_toggle("SPECTRO", vis_tools)
        self.btn_wave = self.create_vis_toggle("WAVE", vis_tools)
        self.btn_pitch = self.create_vis_toggle("PITCH", vis_tools)
        self.btn_harmonics = self.create_vis_toggle("HARMONICS", vis_tools)
        vis_tools.addStretch()
        vis_lay.addLayout(vis_tools)
        
        self.vis = StaticVisualizer(); vis_lay.addWidget(self.vis)
        layout.addWidget(vis_frame)
        
        # --- Header Controls ---
        h_ctrl = QHBoxLayout()
        self.layer_selector = QComboBox(); self.layer_selector.addItems(["LAYER 1", "LAYER 2", "LAYER 3"])
        self.layer_selector.currentIndexChanged.connect(self.update_ui_from_layer)
        h_ctrl.addWidget(self.layer_selector)
        self.layer_active_cb = QCheckBox("ACTIVE"); self.layer_active_cb.stateChanged.connect(self.update_data_from_ui)
        h_ctrl.addWidget(self.layer_active_cb)
        self.note_lbl = QLabel("--")
        h_ctrl.addWidget(self.note_lbl)
        h_ctrl.addStretch()
        
        # Drive Control
        self.drive_spin = QSpinBox(); self.drive_spin.setRange(0, 100); self.drive_spin.valueChanged.connect(self.update_data_from_ui)
        h_ctrl.addWidget(QLabel("Drive:")); h_ctrl.addWidget(self.drive_spin)

        # Master Volume Control
        h_ctrl.addSpacing(20)
        h_ctrl.addWidget(QLabel("Master Vol %:"))
        self.m_vol = QSpinBox()
        self.m_vol.setRange(0, 200)
        self.m_vol.setValue(100)
        self.m_vol.setFixedWidth(60)
        self.m_vol.valueChanged.connect(self.set_master_vol)
        h_ctrl.addWidget(self.m_vol)

        layout.addLayout(h_ctrl)

        # --- Tabbed Interface ---
        tabs = QTabWidget()
        layout.addWidget(tabs)

        # --- TAB 1: SOURCE (Osc, FM, Noise) ---
        t_src = QWidget(); l_src = QHBoxLayout(t_src) # Horizontal for 3 groups
        
        # Oscillator Column
        g_osc = QGroupBox("Oscillator"); v_osc = QVBoxLayout(g_osc)
        self.wave_box = QComboBox(); self.wave_box.addItems(["Sine", "Square", "Triangle", "Sawtooth"]); self.wave_box.currentIndexChanged.connect(self.update_data_from_ui)
        v_osc.addWidget(QLabel("Waveform")); v_osc.addWidget(self.wave_box)
        self.p_base = self.make_slider("Freq (Hz)", 20, 500, 55, v_osc)
        self.p_gain = self.make_slider("Gain %", 0, 150, 100, v_osc)
        l_src.addWidget(g_osc)

        # FM Column
        g_fm = QGroupBox("Frequency Mod"); v_fm = QVBoxLayout(g_fm)
        self.fm_rat = self.make_slider("Ratio (/10)", 1, 200, 10, v_fm)
        self.fm_amt = self.make_slider("Amount", 0, 3000, 0, v_fm)
        v_fm.addStretch()
        l_src.addWidget(g_fm)

        # Noise Column
        g_noise = QGroupBox("Noise Engine"); v_noise = QVBoxLayout(g_noise)
        self.noise_type = QComboBox(); self.noise_type.addItems(["White", "Pink", "Brown"]); self.noise_type.currentIndexChanged.connect(self.update_data_from_ui)
        v_noise.addWidget(QLabel("Type")); v_noise.addWidget(self.noise_type)
        self.n_mix = self.make_slider("Mix %", 0, 100, 0, v_noise)
        v_noise.addStretch()
        l_src.addWidget(g_noise)

        tabs.addTab(t_src, "Source")

        # --- TAB 2: FILTER & FX ---
        t_fx = QWidget(); l_fx = QHBoxLayout(t_fx)

        # Filter Group
        g_filt = QGroupBox("Filter"); v_filt = QVBoxLayout(g_filt)
        self.flt_type_box = QComboBox(); self.flt_type_box.addItems(["Lowpass", "Highpass", "Bandpass"])
        self.flt_type_box.currentIndexChanged.connect(self.update_data_from_ui)
        v_filt.addWidget(QLabel("Mode")); v_filt.addWidget(self.flt_type_box)
        self.n_cut = self.make_slider("Cutoff (Hz)", 20, 18000, 8000, v_filt)
        self.n_q = self.make_slider("Resonance (Q)", 1, 50, 10, v_filt)
        l_fx.addWidget(g_filt)

        # FX Group
        g_eff = QGroupBox("Effects"); v_eff = QVBoxLayout(g_eff)
        self.bit_crush = self.make_slider("Bitcrush", 0, 100, 0, v_eff)
        self.p_spread = self.make_slider("Stereo Spread", 0, 100, 0, v_eff)
        v_eff.addStretch()
        l_fx.addWidget(g_eff)

        tabs.addTab(t_fx, "Filter & FX")

        # --- TAB 3: ENVELOPES ---
        t_env = QWidget(); l_env = QHBoxLayout(t_env)

        # Pitch Env Group
        g_pe = QGroupBox("Pitch Envelope"); v_pe = QVBoxLayout(g_pe)
        self.p_env_int = self.make_slider("Intensity (Hz)", 0, 1500, 0, v_pe)
        self.p_env_dec = self.make_slider("Decay (ms)", 0, 500, 0, v_pe)
        v_pe.addStretch()
        l_env.addWidget(g_pe)

        # Amp Env Group
        g_ae = QGroupBox("Amplitude ADSR"); v_ae = QVBoxLayout(g_ae)
        self.a_att = self.make_slider("Attack (ms)", 0, 100, 0, v_ae)
        self.a_dec = self.make_slider("Decay (ms)", 1, 2000, 100, v_ae)
        self.a_sus = self.make_slider("Sustain %", 0, 100, 0, v_ae)
        self.a_rel = self.make_slider("Release (ms)", 1, 2000, 1, v_ae)
        l_env.addWidget(g_ae)

        tabs.addTab(t_env, "Envelopes")

        # --- Footer ---
        btn = QPushButton("TRIGGER (SPACE)"); btn.setFixedHeight(50); btn.clicked.connect(self.engine.trigger)
        layout.addWidget(btn)
        
        # Action Buttons
        act_lo = QHBoxLayout()
        save_btn = QPushButton("Save Preset"); save_btn.clicked.connect(self.handle_save)
        load_btn = QPushButton("Load Preset"); load_btn.clicked.connect(self.handle_load)
        exp_btn = QPushButton("Export WAV"); exp_btn.clicked.connect(self.handle_export)
        act_lo.addWidget(save_btn); act_lo.addWidget(load_btn); act_lo.addWidget(exp_btn)
        layout.addLayout(act_lo)

    def create_vis_toggle(self, text, lay):
        b = QPushButton(text); b.setObjectName("tgl"); b.setCheckable(True); b.setChecked(True)
        b.clicked.connect(self.update_vis_settings)
        lay.addWidget(b); return b

    def update_vis_settings(self):
        self.vis.show_heat = self.btn_spectro.isChecked()
        self.vis.show_wave = self.btn_wave.isChecked()
        self.vis.show_pitch = self.btn_pitch.isChecked()
        self.vis.show_harmonics = self.btn_harmonics.isChecked()
        self.vis.update()

    def make_slider(self, name, mn, mx, df, lay):
        h = QHBoxLayout()
        h.addWidget(QLabel(name))
        
        # Spinbox
        sp = QSpinBox()
        sp.setRange(mn, mx)
        sp.setValue(df)
        sp.setFixedWidth(60)

        # Slider
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(mn, mx)
        s.setValue(df)
        
        # Sync
        sp.valueChanged.connect(s.setValue)
        s.valueChanged.connect(sp.setValue)
        s.valueChanged.connect(self.update_data_from_ui)
        
        h.addWidget(sp)
        h.addWidget(s)
        lay.addLayout(h)
        return s 

    def update_ui_from_layer(self):
        self.ui_ready = False
        l = self.layers[self.layer_selector.currentIndex()]
        self.layer_active_cb.setChecked(l['active'])
        self.wave_box.setCurrentText(l['waveform_type'])
        self.p_gain.setValue(int(l['gain'] * 100))
        self.p_base.setValue(int(l['base_pitch']))
        self.p_env_int.setValue(int(l['pitch_env_int']))
        self.p_env_dec.setValue(int(l['pitch_env_dec']))
        
        # Filter & FX
        self.flt_type_box.setCurrentText(l.get('filter_type', "Lowpass"))
        self.n_cut.setValue(int(l['noise_cutoff']))
        self.n_q.setValue(int(l.get('noise_q', 1.0)*10))
        self.bit_crush.setValue(int(l.get('bit_crush', 0.0)*100))
        self.p_spread.setValue(int(l.get('spread', 0.0)*100)) # UI Update for Spread
        
        self.fm_rat.setValue(int(l['fm_ratio']*10))
        self.fm_amt.setValue(int(l['fm_amount']))
        self.noise_type.setCurrentText(l['noise_type'])
        self.n_mix.setValue(int(l['noise_mix']*100))
        
        self.a_att.setValue(int(l['amp_attack']))
        self.a_dec.setValue(int(l['amp_decay']))
        self.a_sus.setValue(int(l['amp_sustain']*100))
        self.a_rel.setValue(int(l['amp_release']))
        self.ui_ready = True

    def update_data_from_ui(self):
        if not self.ui_ready: return
        self.drive_val = 1.0 + (self.drive_spin.value() / 10.0)
        l = self.layers[self.layer_selector.currentIndex()]
        l['active'] = self.layer_active_cb.isChecked()
        l['waveform_type'] = self.wave_box.currentText()
        l['gain'] = self.p_gain.value() / 100.0
        l['base_pitch'] = self.p_base.value()
        l['pitch_env_int'] = self.p_env_int.value()
        l['pitch_env_dec'] = self.p_env_dec.value()
        
        l['filter_type'] = self.flt_type_box.currentText()
        l['noise_cutoff'] = self.n_cut.value()
        l['noise_q'] = self.n_q.value() / 10.0
        l['bit_crush'] = self.bit_crush.value() / 100.0
        l['spread'] = self.p_spread.value() / 100.0 # Data Update for Spread
        
        l['fm_ratio'] = self.fm_rat.value() / 10.0
        l['fm_amount'] = self.fm_amt.value()
        l['noise_type'] = self.noise_type.currentText()
        l['noise_mix'] = self.n_mix.value() / 100.0
        
        l['amp_attack'] = self.a_att.value()
        l['amp_decay'] = self.a_dec.value()
        l['amp_sustain'] = self.a_sus.value() / 100.0
        l['amp_release'] = self.a_rel.value()
        
        # Debounce the calculation trigger
        self.recalc_timer.start()

    def get_note_name(self, hz):
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        if hz <= 0: return "--"
        midi = int(round(12 * np.log2(hz / 440.0) + 69))
        return f"{notes[midi % 12]}{ (midi // 12) - 1 }"

    def handle_save(self):
        data = {"drive": self.drive_val, "layers": self.layers}
        path, _ = QFileDialog.getSaveFileName(self, "Save Preset", os.path.join(self.presets_dir, "preset.json"), "JSON (*.json)")
        if path:
            with open(path, 'w') as f: json.dump(data, f, indent=4)

    def handle_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Preset", self.presets_dir, "JSON (*.json)")
        if path:
            try:
                with open(path, 'r') as f: data = json.load(f)
                self.ui_ready = False
                self.drive_val = data.get("drive", 1.0)
                self.drive_spin.setValue(int((self.drive_val - 1.0) * 10))
                self.layers = data["layers"]
                self.ui_ready = True
                self.update_ui_from_layer()
                self.trigger_recalc()
            except Exception as e: print(e)

    def handle_export(self):
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path, _ = QFileDialog.getSaveFileName(self, "Export WAV", os.path.join(self.samples_dir, f"sample_{ts}.wav"), "WAV (*.wav)")
        if path: sf.write(path, self.engine.cached_sample, self.engine.sample_rate)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space: self.engine.trigger()
    
    def closeEvent(self, event):
        self.synth_thread.quit()
        self.synth_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = DrumSynthApp(); win.show(); sys.exit(app.exec())