import cv2
import numpy as np
import os

class PhysicalValidator:
    def __init__(self, output_dir="data/audit_payload"):
        self.output_dir = output_dir
        self.dimensions = ['Lux', 'DPI', 'MBlur', 'Defocus', 'ISO']
        self._prepare_dirs()

    def _prepare_dirs(self):
        """Initialize directory structure for all stress dimensions."""
        for d in self.dimensions:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)

    def _save(self, img, uid, param_name, val):
        """Saves image with standardized naming: UID_Dimension_Value.jpg"""
        filename = f"{uid}_{param_name}_{val:.3f}.jpg" 
        save_path = os.path.join(self.output_dir, param_name, filename)
        cv2.imwrite(save_path, img)

    def apply_illumination(self, img, uid, steps):
        """Simulates lighting conditions using Gamma correction."""
        for alpha in steps:
            inv_gamma = 1.0 / alpha if alpha > 0 else 1.0
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            res = cv2.LUT(img, table)
            self._save(res, uid, "Lux", alpha)

    def apply_resolution(self, img, uid, steps):
        """Simulates spatial sampling loss (DPI degradation)."""
        h, w = img.shape[:2]
        for scale in steps:
            temp = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            res = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            self._save(res, uid, "DPI", scale)

    def apply_motion_blur(self, img, uid, steps):
        """Simulates sub-pixel linear motion blur (Conveyor vibration)."""
        for size in steps:
            if size <= 1: 
                self._save(img, uid, "MBlur", 0)
                continue
            k = int(size) if int(size) % 2 != 0 else int(size) + 1
            kernel = np.zeros((k, k))
            kernel[int((k-1)/2), :] = 1
            kernel /= kernel.sum()
            res = cv2.filter2D(img, -1, kernel)
            self._save(res, uid, "MBlur", size)

    def apply_defocus(self, img, uid, steps):
        """Simulates optical defocusing (Gaussian Blur)."""
        for sigma in steps:
            if sigma == 0:
                self._save(img, uid, "Defocus", 0)
                continue
            res = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            self._save(res, uid, "Defocus", sigma)

    def apply_sensor_noise(self, img, uid, steps):
        """Simulates sensor shot noise (ISO/High Gain)."""
        for intensity in steps:
            if intensity == 0:
                self._save(img, uid, "ISO", 0)
                continue
            noise = np.random.normal(0, intensity, img.shape).astype('float32')
            res = np.clip(img.astype('float32') + noise, 0, 255).astype('uint8')
            self._save(res, uid, "ISO", intensity)