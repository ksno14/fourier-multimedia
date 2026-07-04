import numpy as np
from core.coder import Coder

def test_key_to_seed():
    key1 = "hello_world"
    key2 = "hello_world"
    key3 = "other_key"
    
    assert Coder.key_to_seed(key1) == Coder.key_to_seed(key2)
    assert Coder.key_to_seed(key1) != Coder.key_to_seed(key3)

def test_permutation_lossless():
    np.random.seed(42)
    spectrum = np.random.rand(100) + 1j * np.random.rand(100)
    key = "secure_key_123"
    
    permuted, perm_indices = Coder.permute_spectrum(spectrum, key)
    unpermuted = Coder.unpermute_spectrum(permuted, key)
    
    assert np.allclose(spectrum, unpermuted, atol=1e-12)
    assert not np.allclose(spectrum, permuted, atol=1e-12)

def test_frequency_distances_1d():
    shape = (5,)
    distances = Coder.get_frequency_distances(shape)
    assert distances[2] == 0.0
    assert distances[0] == 1.0
    assert distances[4] == 1.0
    assert np.all(distances >= 0.0) and np.all(distances <= 1.0)

def test_phase_noise_bands():
    np.random.seed(42)
    spectrum = np.ones(100, dtype=complex)
    
    modified = Coder.apply_phase_noise(spectrum, percentage=100.0, max_delta=1.0, band="low", seed=42)
    
    assert np.allclose(np.abs(spectrum), np.abs(modified), atol=1e-12)
    
    distances = Coder.get_frequency_distances((100,))
    for i in range(100):
        d = distances[i]
        if d >= 0.33:
            assert spectrum[i] == modified[i]
