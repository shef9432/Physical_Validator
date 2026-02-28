from core.corruptor import PhysicalValidator
import numpy as np

def run_audit(mode="standard"):
    pv = PhysicalValidator()
    
    # 工业分级定义
    configs = {
        "fast": {  # 快速模式：每种参数只跑 3 档
            "lux": [0.5, 1.0, 1.5],
            "dpi": [1.0, 0.5, 0.2],
            "mblur": [0, 10, 30],
            "defocus": [0, 3],
            "iso": [0, 40]
        },
        "deep_audit": { # 深度审计模式：高密度采样，生成平滑曲线
            "lux": np.linspace(0.1, 1.8, 15),       # 15个光照梯度
            "dpi": np.linspace(1.0, 0.05, 10),     # 10个分辨率梯度
            "mblur": np.linspace(0, 50, 10),       # 10个速度梯度
            "defocus": np.linspace(0, 8, 8),       # 8个失焦梯度
            "iso": np.linspace(0, 100, 8)          # 8个噪声梯度
        }
    }

    conf = configs[mode]
    input_img = "data/input/sample.jpg" # 假设的输入
    uid = "PartA_001"
    
    # 执行精细审计
    img = cv2.imread(input_img)
    pv.apply_illumination(img, uid, conf['lux'])
    pv.apply_resolution(img, uid, conf['dpi'])
    pv.apply_motion_blur(img, uid, conf['mblur'])
    pv.apply_defocus(img, uid, conf['defocus'])
    pv.apply_sensor_noise(img, uid, conf['iso'])

if __name__ == "__main__":
    # 周一在部长面前展示，先开 deep_audit 吓死他（划掉）惊艳他
    run_audit(mode="deep_audit")