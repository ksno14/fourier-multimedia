import numpy as np
import scipy.io.wavfile as wavfile
from typing import Tuple

class Utils:
    """
    Utility helpers for loading/saving audio and generic conversions.
    """

    @staticmethod
    def load_wav(file_path: str) -> Tuple[int, np.ndarray, str]:
        """
        Loads a WAV file and converts it to a normalized float array in [-1.0, 1.0].
        If stereo, downmixes to mono.
        
        Returns:
        - sample_rate: integer sampling rate.
        - data: 1D normalized float array.
        - original_dtype: string indicating the original data type (e.g. 'int16', 'float32').
        """
        sample_rate, data = wavfile.read(file_path)
        original_dtype = str(data.dtype)
        
        # Convert stereo to mono by averaging channels
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
            
        # Normalize to [-1.0, 1.0] based on data type
        if original_dtype.startswith('int16'):
            float_data = data.astype(np.float64) / 32768.0
        elif original_dtype.startswith('int32'):
            float_data = data.astype(np.float64) / 2147483648.0
        elif original_dtype.startswith('uint8'):
            float_data = (data.astype(np.float64) - 128.0) / 128.0
        else:
            # Already float, just copy
            float_data = data.astype(np.float64)
            
        return sample_rate, float_data, original_dtype

    @staticmethod
    def save_wav(file_path: str, data: np.ndarray, sample_rate: int, target_dtype: str = 'int16') -> None:
        """
        Saves a normalized float array as a WAV file, casting back to the target datatype.
        """
        # Clip just in case values exceeded [-1, 1] during processing
        clipped_data = np.clip(data, -1.0, 1.0)
        
        if target_dtype.startswith('int16'):
            out_data = (clipped_data * 32767.0).astype(np.int16)
        elif target_dtype.startswith('int32'):
            out_data = (clipped_data * 2147483647.0).astype(np.int32)
        elif target_dtype.startswith('uint8'):
            out_data = ((clipped_data * 127.0) + 128.0).astype(np.uint8)
        else:
            out_data = clipped_data.astype(np.float32)
            
        wavfile.write(file_path, sample_rate, out_data)
