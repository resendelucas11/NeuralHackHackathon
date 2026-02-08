import cv2
import numpy as np


class ROIMaskService:
    @staticmethod
    def polygon_mask(image_h: int, image_w: int, points_xy):
        """
        points_xy: lista de (x,y) en píxeles
        devuelve máscara 0/255
        """
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        if points_xy and len(points_xy) >= 3:
            pts = np.array(points_xy, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        return mask

    @staticmethod
    def overlay_mask(img_bgr: np.ndarray, mask_0_255: np.ndarray, alpha: float = 0.35, color_bgr=(0, 0, 255)):
        """
        Pinta la máscara encima de la imagen (segmento coloreado).
        color_bgr=(0,0,255) -> rojo
        """
        overlay = img_bgr.copy()
        colored = np.zeros_like(img_bgr, dtype=np.uint8)
        colored[:] = color_bgr
        m = mask_0_255 > 0
        overlay[m] = (overlay[m] * (1 - alpha) + colored[m] * alpha).astype(np.uint8)
        return overlay

    @staticmethod
    def draw_polygon_edges(img_bgr: np.ndarray, points_xy, color_bgr=(0, 255, 0), thickness=3):
        if points_xy and len(points_xy) >= 2:
            pts = np.array(points_xy, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_bgr, [pts], isClosed=True, color=color_bgr, thickness=thickness)
        return img_bgr

    @staticmethod
    def bounding_rect(points_xy):
        xs = [p[0] for p in points_xy]
        ys = [p[1] for p in points_xy]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        return [int(x1), int(y1), int(x2), int(y2)]
