import numpy as np
from PIL import Image
from core.fourier import FourierProcessor
from core.coder import Coder
from core.metrics import Metrics

def load_image_gray(file_path: str) -> np.ndarray:
    """Loads an image and converts it to a 2D numpy array in grayscale."""
    img = Image.open(file_path).convert('L')
    return np.array(img, dtype=np.float64)

def process_image(img_gray: np.ndarray, mode: str, params: dict) -> dict:
    """
    Processes a grayscale image through the 2D Fourier encoding/decoding pipeline.
    
    Parameters:
    - img_gray: 2D numpy array of the grayscale image.
    - mode: "permutation" or "phase".
    - params: Coder settings ("key" for permutation; "percentage", "max_delta", "band" for phase).
    
    Returns:
    - Dictionary with original, coded, decoded images, spectrums, and metrics (MSE, PSNR, SSIM).
    """
    # 1. FFT2
    spectrum = FourierProcessor.fft2(img_gray)
    
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
        
        # Apply phase noise on the shifted 2D spectrum
        shifted_spectrum = FourierProcessor.fft_shift(spectrum)
        modified_shifted = Coder.apply_phase_noise(shifted_spectrum, percentage, max_delta, band)
        
        # Unshift back
        coded_spectrum = FourierProcessor.ifft_shift(modified_shifted)
        decoded_spectrum = coded_spectrum.copy()
        
    else:
        coded_spectrum = spectrum.copy()
        decoded_spectrum = spectrum.copy()
        
    # 3. IFFT2 for coded (distorted/encrypted) image
    coded_image = np.real(FourierProcessor.ifft2(coded_spectrum))
    coded_image_clipped = np.clip(coded_image, 0.0, 255.0)
    
    # 4. IFFT2 for decoded (recovered) image
    decoded_image = np.real(FourierProcessor.ifft2(decoded_spectrum))
    decoded_image_clipped = np.clip(decoded_image, 0.0, 255.0)
    
    # Calculate 2D metrics (between original and decoded)
    mse = Metrics.mse(img_gray, decoded_image_clipped)
    psnr = Metrics.psnr(img_gray, decoded_image_clipped, max_val=255.0)
    ssim = Metrics.ssim(img_gray, decoded_image_clipped, max_val=255.0)
    
    # Generate log-magnitude for visualization
    magnitude = FourierProcessor.magnitude(spectrum)
    shifted_magnitude = FourierProcessor.fft_shift(magnitude)
    log_magnitude = np.log1p(shifted_magnitude)
    
    # Log magnitude of coded spectrum
    coded_magnitude = FourierProcessor.magnitude(coded_spectrum)
    coded_shifted_magnitude = FourierProcessor.fft_shift(coded_magnitude)
    coded_log_magnitude = np.log1p(coded_shifted_magnitude)
    
    phase = FourierProcessor.phase(spectrum)
    shifted_phase = FourierProcessor.fft_shift(phase)
    
    return {
        "original_image": img_gray,
        "original_spectrum": spectrum,
        "log_magnitude": log_magnitude,
        "phase": shifted_phase,
        
        "coded_image": coded_image_clipped,
        "coded_spectrum": coded_spectrum,
        "coded_log_magnitude": coded_log_magnitude,
        
        "decoded_image": decoded_image_clipped,
        "decoded_spectrum": decoded_spectrum,
        
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim
    }

def extract_edges(img_gray: np.ndarray, cutoff_fraction: float = 0.05) -> np.ndarray:
    """
    Extracts image edges using a high-pass filter in the frequency domain.
    
    Parameters:
    - img_gray: 2D numpy array of the grayscale image.
    - cutoff_fraction: Fraction of the frequency space to cut off from the center (0.0 to 1.0).
    """
    # 1. FFT2
    spectrum = FourierProcessor.fft2(img_gray)
    shifted_spectrum = FourierProcessor.fft_shift(spectrum)
    
    # 2. Get frequency distances from center
    distances = Coder.get_frequency_distances(img_gray.shape)
    
    # 3. Create high-pass filter mask (block low frequencies)
    high_pass_mask = distances >= cutoff_fraction
    
    # 4. Apply mask
    filtered_shifted = shifted_spectrum * high_pass_mask
    
    # 5. IFFT2
    filtered_spectrum = FourierProcessor.ifft_shift(filtered_shifted)
    edges = np.real(FourierProcessor.ifft2(filtered_spectrum))
    
    # 6. Normalize output for visualization
    edges_abs = np.abs(edges)
    min_val = np.min(edges_abs)
    max_val = np.max(edges_abs)
    
    if max_val - min_val > 1e-5:
        edges_normalized = ((edges_abs - min_val) / (max_val - min_val)) * 255.0
    else:
        edges_normalized = np.zeros_like(edges_abs)
        
    return edges_normalized.astype(np.uint8)
