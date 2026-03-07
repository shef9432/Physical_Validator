import cv2
import numpy as np

class StressSimulator:
    """Physics-based degradation operators for industrial environments."""
    
    @staticmethod
    def apply_lux_plus(img: np.ndarray, val: int) -> np.ndarray:
        return np.clip(img.astype(np.float32) * (1 + val / 50), 0, 255).astype(np.uint8)

    @staticmethod
    def apply_mblur(img: np.ndarray, val: int) -> np.ndarray:
        kernel_size = int(val / 10) * 2 + 1
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) if kernel_size > 1 else img

    @staticmethod
    def apply_iso_noise(img: np.ndarray, val: int) -> np.ndarray:
        noise = np.random.normal(0, val, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)