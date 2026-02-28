import cv2
import numpy as np
import os
from core.corruptor import PhysicalValidator
from utils.logger import AuditLogger

def run_deep_audit():
    input_file = "data/input/sample.jpg"
    if not os.path.exists(input_file):
        print(f"❌ Input image not found: {input_file}")
        return

    img = cv2.imread(input_file)
    pv = PhysicalValidator()
    logger = AuditLogger()
    uid = "PROJ_7432_UNIT_01"

    print("🔥 Starting multi-dimensional high-resolution audit generation...")

    # Illumination (Lux): 50 samples
    pv.apply_illumination(img, uid, np.linspace(0.05, 3.0, 50))
    # Resolution (DPI): 30 samples
    pv.apply_resolution(img, uid, np.linspace(0.1, 1.0, 30))
    # Motion Blur (MBlur): 40 samples
    pv.apply_motion_blur(img, uid, np.linspace(1, 100, 40))
    # Defocus Loss: 40 samples
    pv.apply_defocus(img, uid, np.linspace(0, 15, 40))
    # Sensor Noise (ISO): 40 samples
    pv.apply_sensor_noise(img, uid, np.linspace(0, 120, 40))

    logger.build_template()
    print("✨ Physical stress-test samples and audit list are ready.")

if __name__ == "__main__":
    run_deep_audit()