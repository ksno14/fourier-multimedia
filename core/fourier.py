import numpy as np

class FourierProcessor:
    """
    Mathematical processor for Discrete Fourier Transforms (DFT/FFT)
    supporting both 1D (text/audio) and 2D (image) signals.
    """

    @staticmethod
    def fft(signal: np.ndarray) -> np.ndarray:
        """Computes the 1D Fast Fourier Transform."""
        return np.fft.fft(signal)

    @staticmethod
    def ifft(spectrum: np.ndarray) -> np.ndarray:
        """Computes the 1D Inverse Fast Fourier Transform."""
        return np.fft.ifft(spectrum)

    @staticmethod
    def fft2(image: np.ndarray) -> np.ndarray:
        """Computes the 2D Fast Fourier Transform."""
        return np.fft.fft2(image)

    @staticmethod
    def ifft2(spectrum: np.ndarray) -> np.ndarray:
        """Computes the 2D Inverse Fast Fourier Transform."""
        return np.fft.ifft2(spectrum)

    @staticmethod
    def magnitude(spectrum: np.ndarray) -> np.ndarray:
        """Extracts the amplitude (magnitude) of the Fourier spectrum."""
        return np.abs(spectrum)

    @staticmethod
    def phase(spectrum: np.ndarray) -> np.ndarray:
        """Extracts the phase (angles in radians) of the Fourier spectrum."""
        return np.angle(spectrum)

    @staticmethod
    def build_spectrum(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Reconstructs the complex spectrum from magnitude and phase."""
        return magnitude * np.exp(1j * phase)

    @staticmethod
    def fft_shift(spectrum: np.ndarray) -> np.ndarray:
        """Shifts the zero-frequency component to the center of the spectrum."""
        return np.fft.fftshift(spectrum)

    @staticmethod
    def ifft_shift(spectrum: np.ndarray) -> np.ndarray:
        """Reverses the zero-frequency shift back to standard ordering."""
        return np.fft.ifftshift(spectrum)
