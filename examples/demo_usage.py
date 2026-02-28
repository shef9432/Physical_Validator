from core.corruptor import PhysicalValidator
from utils.logger import AuditLogger
from utils.visualizer import PhysicalVisualizer
import numpy as np

# 1. 物理审计配置 (最高精度模式)
pv = PhysicalValidator(output_dir="data/audit_payload")
logger = AuditLogger()
viz = PhysicalVisualizer()

# 定义高精度步长 (linspace 保证了曲线的平滑)
high_res_config = {
    "lux": np.linspace(0.1, 1.5, 20),      # 20步光照模拟
    "mblur": np.linspace(0, 60, 15),       # 15步运动模糊模拟
    "defocus": np.linspace(0, 10, 15)      # 15步失焦模拟
}

# 2. 生成“物理病毒”
# 假设你 data/input 里已经放了一张图叫 "gear_sample.jpg"
pv.process_any("data/input/gear_sample.jpg") 

# 3. 生成 CSV 模板
logger.build_template()

print("\n--- 预测试完成 ---")
print("现在的任务：")
print("1. 查看 data/audit_payload 下的图片，确认物理退化效果。")
print("2. (手动模拟) 在 CSV 里填入一些随机的 Confidence 数值，测试绘图效果。")
print("3. 运行 viz.plot_robustness() 看看你的曲线图！")