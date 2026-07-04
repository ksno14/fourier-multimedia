import numpy as np
from core.fourier import FourierProcessor

def test_fft_ifft_1d():
    t = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 12 * t)
    
    spectrum = FourierProcessor.fft(signal)
    reconstructed = FourierProcessor.ifft(spectrum)
    
    assert np.allclose(signal, np.real(reconstructed), atol=1e-12)

def test_fft_ifft_2d():
    np.random.seed(42)
    image = np.random.rand(32, 32) * 255.0
    
    spectrum = FourierProcessor.fft2(image)
    reconstructed = FourierProcessor.ifft2(spectrum)
    
    assert np.allclose(image, np.real(reconstructed), atol=1e-12)

def test_magnitude_phase_reconstruction():
    np.random.seed(42)
    signal = np.random.rand(50)
    
    spectrum = FourierProcessor.fft(signal)
    
    mag = FourierProcessor.magnitude(spectrum)
    ph = FourierProcessor.phase(spectrum)
    
    rebuilt_spectrum = FourierProcessor.build_spectrum(mag, ph)
    
    assert np.allclose(spectrum, rebuilt_spectrum, atol=1e-12)

def test_shifts():
    spectrum = np.array([1, 2, 3, 4, 5])
    shifted = FourierProcessor.fft_shift(spectrum)
    unshifted = FourierProcessor.ifft_shift(shifted)
    
    assert np.allclose(spectrum, unshifted)
