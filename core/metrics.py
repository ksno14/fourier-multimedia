import numpy as np
from scipy.ndimage import uniform_filter

class Metrics:
    """
    Implements analytical comparison metrics for 1D and 2D signals.
    """

    @staticmethod
    def rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Computes Root Mean Squared Error between original and reconstructed signals."""
        return float(np.sqrt(np.mean((original - reconstructed) ** 2)))

    @staticmethod
    def snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Computes Signal-to-Noise Ratio in decibels (dB).
        Returns infinity if the reconstruction is mathematically identical.
        """
        noise = original - reconstructed
        sum_sq_original = np.sum(original.astype(np.float64) ** 2)
        sum_sq_noise = np.sum(noise.astype(np.float64) ** 2)
        
        if sum_sq_noise == 0:
            return float('inf')
        if sum_sq_original == 0:
            # If original is all zeros and noise is non-zero, SNR is negative infinity or 0
            return -float('inf')
            
        return float(10 * np.log10(sum_sq_original / sum_sq_noise))

    @staticmethod
    def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Computes Mean Squared Error (MSE) between two arrays."""
        return float(np.mean((original - reconstructed) ** 2))

    @staticmethod
    def psnr(original: np.ndarray, reconstructed: np.ndarray, max_val: float = 255.0) -> float:
        """
        Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
        Returns infinity if the reconstruction is mathematically identical.
        """
        mse_val = Metrics.mse(original, reconstructed)
        if mse_val == 0:
            return float('inf')
        return float(20 * np.log10(max_val / np.sqrt(mse_val)))

    @staticmethod
    def ssim(img1: np.ndarray, img2: np.ndarray, size: int = 7, max_val: float = 255.0) -> float:
        """
        Computes the Structural Similarity Index Measure (SSIM) between two grayscale images.
        Uses a uniform sliding window filter for efficiency and local comparison.
        
        Parameters:
        - img1: Grayscale original image (numpy 2D array).
        - img2: Grayscale reconstructed image (numpy 2D array).
        - size: Sliding window size (default 7).
        - max_val: Dynamic range of the pixel values (default 255.0).
        """
        # Convert to float64 to prevent overflow
        x = img1.astype(np.float64)
        y = img2.astype(np.float64)
        
        # Stability constants
        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        
        # Local means using uniform filter
        ux = uniform_filter(x, size)
        uy = uniform_filter(y, size)
        
        # Local variances and covariances
        uxx = uniform_filter(x * x, size)
        uyy = uniform_filter(y * y, size)
        uxy = uniform_filter(x * y, size)
        
        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy
        
        # Ensure non-negative variance values due to numerical precision
        vx = np.maximum(0.0, vx)
        vy = np.maximum(0.0, vy)
        
        # SSIM map formula
        num = (2 * ux * uy + c1) * (2 * vxy + c2)
        den = (ux**2 + uy**2 + c1) * (vx + vy + c2)
        
        ssim_map = num / den
        return float(np.mean(ssim_map))
