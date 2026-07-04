import numpy as np
from core.fourier import FourierProcessor
from core.coder import Coder
from core.metrics import Metrics

def text_to_ascii(text: str) -> np.ndarray:
    """Converts a text string to a float numpy array of ASCII values."""
    return np.array([ord(char) for char in text], dtype=np.float64)

def ascii_to_text(ascii_array: np.ndarray) -> str:
    """Converts a numpy array of ASCII values back to a string, with rounding and clamping."""
    rounded = np.clip(np.round(ascii_array), 0, 255).astype(int)
    chars = []
    for val in rounded:
        try:
            chars.append(chr(val))
        except ValueError:
            chars.append('?')
    return "".join(chars)

def process_text(text: str, mode: str, params: dict) -> dict:
    """
    Processes a text string through the full Fourier encoding/decoding pipeline.
    
    Parameters:
    - text: The input string.
    - mode: "permutation" or "phase".
    - params: Dictionary containing parameters ("key" for permutation; "percentage", "max_delta", "band" for phase).
    
    Returns:
    - Dictionary with original, coded, and decoded signals, spectrums, and error metrics.
    """
    if not text:
        text = " "
        
    ascii_signal = text_to_ascii(text)
    
    # 1. FFT
    spectrum = FourierProcessor.fft(ascii_signal)
    
    # Initialize variables
    coded_spectrum = None
    decoded_spectrum = None
    
    # 2. Coder
    if mode == "permutation":
        key = params.get("key", "default_key")
        coded_spectrum, _ = Coder.permute_spectrum(spectrum, key)
        decoded_spectrum = Coder.unpermute_spectrum(coded_spectrum, key)
        
    elif mode == "phase":
        percentage = params.get("percentage", 10.0)
        max_delta = params.get("max_delta", np.pi)
        band = params.get("band", "all")
        key = params.get("key", "default_key")
        
        # Apply phase noise on the shifted spectrum for proper band targeting
        shifted_spectrum = FourierProcessor.fft_shift(spectrum)
        modified_shifted = Coder.apply_phase_noise(shifted_spectrum, percentage, max_delta, band, key)
        
        coded_spectrum = FourierProcessor.ifft_shift(modified_shifted)
        
        # Decode by removing the phase noise on the shifted spectrum
        decoded_shifted = Coder.remove_phase_noise(modified_shifted, percentage, max_delta, band, key)
        decoded_spectrum = FourierProcessor.ifft_shift(decoded_shifted)
        
    else:
        coded_spectrum = spectrum.copy()
        decoded_spectrum = spectrum.copy()
        
    # 3. IFFT for coded (encrypted/distorted) signal
    coded_signal = np.real(FourierProcessor.ifft(coded_spectrum))
    coded_text = ascii_to_text(coded_signal)
    
    # 4. IFFT for decoded (recovered) signal
    decoded_signal = np.real(FourierProcessor.ifft(decoded_spectrum))
    decoded_text = ascii_to_text(decoded_signal)
    
    # Calculate metrics between original ASCII and recovered ASCII
    rmse = Metrics.rmse(ascii_signal, decoded_signal)
    snr = Metrics.snr(ascii_signal, decoded_signal)
    error = decoded_signal - ascii_signal
    
    return {
        "original_ascii": ascii_signal,
        "original_spectrum": spectrum,
        "magnitude": FourierProcessor.magnitude(spectrum),
        "phase": FourierProcessor.phase(spectrum),
        
        "coded_spectrum": coded_spectrum,
        "coded_ascii": coded_signal,
        "coded_text": coded_text,
        
        "decoded_spectrum": decoded_spectrum,
        "decoded_ascii": decoded_signal,
        "decoded_text": decoded_text,
        
        "error": error,
        "rmse": rmse,
        "snr": snr
    }
