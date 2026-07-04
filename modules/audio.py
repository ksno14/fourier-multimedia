import numpy as np
from core.fourier import FourierProcessor
from core.coder import Coder
from core.metrics import Metrics
from core.utils import Utils

def process_audio(input_wav_path: str, mode: str, params: dict) -> dict:
    """
    Loads, processes (FFT, Encodes/Decodes, IFFT), and calculates metrics for a WAV file.
    
    Parameters:
    - input_wav_path: Path to the input WAV file.
    - mode: "permutation" or "phase".
    - params: Coder settings ("key" for permutation; "percentage", "max_delta", "band" for phase).
    
    Returns:
    - Dictionary with original, coded, decoded waveforms, spectrums, and metrics.
    """
    # Load and normalize audio
    sample_rate, audio_signal, original_dtype = Utils.load_wav(input_wav_path)
    
    # 1. FFT
    spectrum = FourierProcessor.fft(audio_signal)
    
    coded_spectrum = None
    decoded_spectrum = None
    
    # 2. Scramble / Encode
    if mode == "permutation":
        key = params.get("key", "default_key")
        coded_spectrum, _ = Coder.permute_spectrum(spectrum, key)
        decoded_spectrum = Coder.unpermute_spectrum(coded_spectrum, key)
        
    elif mode == "phase":
        percentage = params.get("percentage", 10.0)
        max_delta = params.get("max_delta", np.pi)
        band = params.get("band", "all")
        key = params.get("key", "default_key")
        
        # Apply phase noise on the shifted spectrum
        shifted_spectrum = FourierProcessor.fft_shift(spectrum)
        modified_shifted = Coder.apply_phase_noise(shifted_spectrum, percentage, max_delta, band, key)
        
        coded_spectrum = FourierProcessor.ifft_shift(modified_shifted)
        
        # Decode by removing phase noise on the shifted spectrum
        decoded_shifted = Coder.remove_phase_noise(modified_shifted, percentage, max_delta, band, key)
        decoded_spectrum = FourierProcessor.ifft_shift(decoded_shifted)
        
    else:
        coded_spectrum = spectrum.copy()
        decoded_spectrum = spectrum.copy()
        
    # 3. IFFT
    coded_signal = np.real(FourierProcessor.ifft(coded_spectrum))
    decoded_signal = np.real(FourierProcessor.ifft(decoded_spectrum))
    
    # Calculate audio quality metrics (SNR, RMSE)
    rmse = Metrics.rmse(audio_signal, decoded_signal)
    snr = Metrics.snr(audio_signal, decoded_signal)
    
    return {
        "sample_rate": sample_rate,
        "original_signal": audio_signal,
        "original_spectrum": spectrum,
        "magnitude": FourierProcessor.magnitude(spectrum),
        "phase": FourierProcessor.phase(spectrum),
        
        "coded_signal": coded_signal,
        "coded_spectrum": coded_spectrum,
        
        "decoded_signal": decoded_signal,
        "decoded_spectrum": decoded_spectrum,
        
        "rmse": rmse,
        "snr": snr,
        "original_dtype": original_dtype
    }
