from ultralytics import YOLO
import numpy as np

class YoloService:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, img_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.7):
        # Devuelve el objeto Results de ultralytics
        results = self.model.predict(img_bgr, conf=conf, iou=iou, verbose=False)
        return results[0]  # una imagen => un Results
