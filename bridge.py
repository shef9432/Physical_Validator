import os
import pandas as pd
import json
from utils.logger import AuditLogger

class UniversalAuditor:
    def __init__(self):
        self.baseline_data = None # Storage for original image results

    def get_inference(self, image_path):
        """
        🚀 THE ADAPTATION LAYER
        This is where the user connects their specific algorithm.
        For GitHub: This stays as a stub.
        For Company: Call the Customer's API/SDK/EXE here.
        """
        # Example structure returned by any vision system:
        # return [{"box": [10, 10, 50, 50], "label": 0, "conf": 0.9}]
        raise NotImplementedError("Connect your model's inference call here.")

    def calculate_iou(self, boxA, boxB):
        xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def run_audit(self, original_img, audit_dir):
        # 1. Capture Baseline from the "Perfect" Image
        print(f"--- Establishing Baseline from Original ---")
        self.baseline_data = self.get_inference(original_img)
        
        # 2. Compare against degraded versions
        logger = AuditLogger()
        df = pd.read_csv(logger.report_path)
        
        relative_scores = []
        for _, row in df.iterrows():
            img_path = os.path.join(audit_dir, row['Image_Path'])
            current_preds = self.get_inference(img_path)
            
            # Find the best match for each baseline object
            best_conf = 0.0
            for b_obj in self.baseline_data:
                for p_obj in current_preds:
                    iou = self.calculate_iou(b_obj['box'], p_obj['box'])
                    if iou > 0.5 and b_obj['label'] == p_obj['label']:
                        best_conf = max(best_conf, p_obj['conf'])
            relative_scores.append(best_conf)
            
        df['Validated_Conf'] = relative_scores
        logger.update_results(df)
