import os
import pandas as pd

class AuditLogger:
    def __init__(self, payload_dir="data/audit_payload", report_path="data/audit_report.csv"):
        self.payload_dir = payload_dir
        self.report_path = report_path

    def build_template(self):
        """Scans payload directories and creates the initial audit CSV."""
        rows = []
        dimensions = ['Lux', 'DPI', 'MBlur', 'Defocus', 'ISO']
        
        for dim in dimensions:
            dim_path = os.path.join(self.payload_dir, dim)
            if not os.path.exists(dim_path): continue
            
            files = sorted(os.listdir(dim_path))
            for fname in files:
                if fname.endswith('.jpg'):
                    parts = os.path.splitext(fname)[0].split('_')
                    if len(parts) >= 3:
                        rows.append({
                            "UID": parts[0],
                            "Category": dim,
                            "Param_Value": float(parts[-1]),
                            "Image_Name": fname,
                            "Confidence": 0.0,
                            "AI_Status": "Pending"
                        })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(self.report_path, index=False, encoding='utf-8-sig')
            print(f"📋 Audit template generated: {len(df)} samples registered.")

    def update_result(self, image_name, confidence, status):
        """Updates AI inference results for a specific sample."""
        if not os.path.exists(self.report_path): return
        df = pd.read_csv(self.report_path)
        mask = df['Image_Name'] == image_name
        if mask.any():
            df.loc[mask, 'Confidence'] = confidence
            df.loc[mask, 'AI_Status'] = status
            df.to_csv(self.report_path, index=False, encoding='utf-8-sig')