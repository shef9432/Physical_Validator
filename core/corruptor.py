import cv2
import numpy as np
import os

class PhysicalValidator:
    def __init__(self, output_dir="data/audit_payload"):
        self.output_dir = output_dir
        self.dimensions = ['Lux', 'DPI', 'MBlur', 'Defocus', 'ISO']
        self._prepare_dirs()

    def _prepare_dirs(self):
        for d in self.dimensions:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)

    def _save(self, img, uid, param_name, val):
        filename = f"{uid}_{param_name}_{val:.3f}.jpg" # 提高命名精度到三位小数
        save_path = os.path.join(self.output_dir, param_name, filename)
        cv2.imwrite(save_path, img)

    # --- 高精度物理演化算法 ---

    def apply_illumination(self, img, uid, steps):
        """线性模拟光照细微变化"""
        for alpha in steps:
            # 使用伽马校正(Gamma)模拟更真实的亮度变化而非线性增益
            # res = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            inv_gamma = 1.0 / alpha if alpha > 0 else 1.0
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            res = cv2.LUT(img, table)
            self._save(res, uid, "Lux", alpha)

    def apply_resolution(self, img, uid, steps):
        """模拟空间采样率丢失"""
        h, w = img.shape[:2]
        for scale in steps:
            temp = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            res = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            self._save(res, uid, "DPI", scale)

    def apply_motion_blur(self, img, uid, steps):
        """模拟亚像素级位移模糊"""
        for size in steps:
            if size <= 1: 
                self._save(img, uid, "MBlur", 0)
                continue
            kernel = np.zeros((int(size), int(size)))
            kernel[int((size-1)/2), :] = 1
            kernel /= kernel.sum()
            res = cv2.filter2D(img, -1, kernel)
            self._save(res, uid, "MBlur", size)

    def apply_defocus(self, img, uid, steps):
        """模拟镜头失焦 (光学审计的核心)"""
        for sigma in steps:
            if sigma == 0:
                self._save(img, uid, "Defocus", 0)
                continue
            # 使用高斯模糊模拟弥散圆
            res = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            self._save(res, uid, "Defocus", sigma)

    def apply_sensor_noise(self, img, uid, steps):
        """模拟传感器散粒噪声 (Shot Noise)"""
        for intensity in steps:
            if intensity == 0:
                self._save(img, uid, "ISO", 0)
                continue
            # 模拟更真实的泊松噪声分布
            noise = np.random.normal(0, intensity, img.shape).astype('float32')
            res = np.clip(img.astype('float32') + noise, 0, 255).astype('uint8')
            self._save(res, uid, "ISO", intensity)