import hashlib
import numpy as np
from typing import Tuple, Union, Optional

class Coder:
    """
    Implements signal encoding and decoding algorithms in the frequency domain.
    Supports Permutation Mode (lossless scrambling) and Phase Mode (frequency band-specific noise injection).
    """

    @staticmethod
    def key_to_seed(key: str) -> int:
        """Converts a string key to a 32-bit integer seed using SHA-256."""
        sha256_hash = hashlib.sha256(key.encode()).hexdigest()
        # Take the first 8 characters (32 bits) and convert to integer
        return int(sha256_hash[:8], 16)

    @classmethod
    def permute_spectrum(cls, spectrum: np.ndarray, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Permutes the complex coefficients of a 1D or 2D spectrum based on a string key.
        Returns the permuted spectrum and the permutation index array.
        """
        seed = cls.key_to_seed(key)
        rng = np.random.default_rng(seed)
        
        original_shape = spectrum.shape
        flat_spectrum = spectrum.flatten()
        n = len(flat_spectrum)
        
        permutation = rng.permutation(n)
        permuted_flat = flat_spectrum[permutation]
        
        return permuted_flat.reshape(original_shape), permutation

    @classmethod
    def unpermute_spectrum(cls, permuted_spectrum: np.ndarray, key: str) -> np.ndarray:
        """
        Inverts the permutation of the complex coefficients to recover the original spectrum.
        """
        seed = cls.key_to_seed(key)
        rng = np.random.default_rng(seed)
        
        original_shape = permuted_spectrum.shape
        flat_permuted = permuted_spectrum.flatten()
        n = len(flat_permuted)
        
        permutation = rng.permutation(n)
        inverse_permutation = np.argsort(permutation)
        
        unpermuted_flat = flat_permuted[inverse_permutation]
        return unpermuted_flat.reshape(original_shape)

    @staticmethod
    def get_frequency_distances(shape: Tuple[int, ...]) -> np.ndarray:
        """
        Calculates normalized frequency distances from the center (DC) component for a shifted spectrum.
        Returns an array of the same shape with distances normalized between 0.0 and 1.0.
        """
        if len(shape) == 1:
            n = shape[0]
            center = n // 2
            indices = np.arange(n)
            # Distance from center normalized to [0, 1]
            max_dist = max(center, n - 1 - center)
            if max_dist == 0:
                return np.zeros(shape)
            return np.abs(indices - center) / max_dist
            
        elif len(shape) == 2:
            h, w = shape
            r_center, c_center = h // 2, w // 2
            
            # Create coordinate grids
            r_indices = np.arange(h)
            c_indices = np.arange(w)
            r_grid, c_grid = np.meshgrid(r_indices, c_indices, indexing='ij')
            
            # Normalized distances along each axis
            r_dist = (r_grid - r_center) / max(r_center, 1)
            c_dist = (c_grid - c_center) / max(c_center, 1)
            
            # Euclidean distance, normalized to [0, 1] at corners
            dist = np.sqrt(r_dist**2 + c_dist**2) / np.sqrt(2.0)
            return np.clip(dist, 0.0, 1.0)
            
        else:
            raise ValueError("Only 1D and 2D arrays are supported.")

    @classmethod
    def apply_phase_noise(
        cls, 
        spectrum: np.ndarray, 
        percentage: float, 
        max_delta: float, 
        band: str,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """
        Modifies the phase of a configurable subset of spectrum coefficients in a specific frequency band.
        
        Parameters:
        - spectrum: Complex Fourier spectrum (should be fft_shifted for correct band selection).
        - percentage: Percentage of coefficients in the band to modify (0 to 100).
        - max_delta: Maximum phase shift in radians.
        - band: Frequency band to target ("low", "mid", "high", "all").
        - seed: Random seed for reproducibility.
        """
        # Get normalized frequency distances
        distances = cls.get_frequency_distances(spectrum.shape)
        
        # Define band masks
        if band == "low":
            mask = (distances >= 0.0) & (distances < 0.33)
        elif band == "mid":
            mask = (distances >= 0.33) & (distances < 0.66)
        elif band == "high":
            mask = (distances >= 0.66) & (distances <= 1.0)
        elif band == "all":
            mask = np.ones(spectrum.shape, dtype=bool)
        else:
            raise ValueError(f"Unknown frequency band: {band}")
            
        # Get flat indices where mask is True
        candidate_indices = np.where(mask.flatten())[0]
        n_candidates = len(candidate_indices)
        
        if n_candidates == 0:
            return spectrum.copy()
            
        # Select subset of candidate indices to scramble
        rng = np.random.default_rng(seed)
        num_to_modify = int(np.round((percentage / 100.0) * n_candidates))
        
        if num_to_modify == 0:
            return spectrum.copy()
            
        selected_flat_indices = rng.choice(candidate_indices, size=num_to_modify, replace=False)
        
        # Scramble phase
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        
        # Flatten for easy index modification
        flat_phase = phase.flatten()
        
        # Draw phase deltas in [-max_delta, max_delta]
        phase_deltas = rng.uniform(-max_delta, max_delta, size=num_to_modify)
        
        # Apply deltas
        flat_phase[selected_flat_indices] += phase_deltas
        
        # Reconstruct complex values
        modified_phase = flat_phase.reshape(spectrum.shape)
        
        # Build new complex spectrum
        modified_spectrum = magnitude * np.exp(1j * modified_phase)
        return modified_spectrum
