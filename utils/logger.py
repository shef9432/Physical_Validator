import os
import pandas as pd

class AuditLogger:
    def __init__(self, payload_dir="data/audit_payload", report_path="data/audit_report.csv"):
        self.payload_dir = payload_dir
        self.report_path = report_path

    def build_template(self):
        """扫描所有子文件夹，生成精确的 CSV 审计模板"""
        rows = []
        # 对应我们在 corruptor 中定义的维度
        dimensions = ['Lux', 'DPI', 'MBlur', 'Defocus', 'ISO']
        
        for dim in dimensions:
            dim_path = os.path.join(self.payload_dir, dim)
            if not os.path.exists(dim_path): continue
            
            for fname in os.listdir(dim_path):
                if fname.endswith(('.jpg', '.png', '.bmp')):
                    # 解析文件名：UID_Param_Value.jpg
                    # 例如：PartA_Lux_0.357.jpg
                    name_parts = os.path.splitext(fname)[0].split('_')
                    if len(name_parts) >= 3:
                        uid = name_parts[0]
                        val = name_parts[-1] # 取最后一个部分作为数值
                        
                        rows.append({
                            "UID": uid,
                            "Category": dim,
                            "Param_Value": float(val),
                            "Image_Path": os.path.join(dim, fname),
                            "AI_Status": "Pending", # 状态：Pending/Success/Fail
                            "Confidence": 0.0,       # 核心 Y 轴数据
                            "Inference_Time": 0.0    # 可选：模拟产线节拍
                        })
        
        df = pd.DataFrame(rows)
        # 按照类别和数值排序，确保绘图时是连续的
        df = df.sort_values(by=['Category', 'Param_Value'])
        df.to_csv(self.report_path, index=False, encoding='utf-8-sig')
        print(f"📋 审计报告模板已生成: {self.report_path} (共 {len(df)} 条测试用例)")
        
        def update_result(self, image_name, confidence, status):
        """外部代码调用此函数回填 AI 推理结果"""
        df = pd.read_csv(self.report_path)
        df.loc[df['Image_Name'] == image_name, 'Confidence'] = confidence
        df.loc[df['Image_Name'] == image_name, 'AI_Status'] = status
        df.to_csv(self.report_path, index=False)