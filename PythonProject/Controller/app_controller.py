import os
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np

from Model.yolo_service import YoloService
from Model.metrics_service import MetricsService
from Model.evidence_service import EvidenceService
from Model.roi_mask_service import ROIMaskService


class AppController:
    def __init__(self, model_path: str, outputs_dir: str = "outputs"):
        self.model_path = model_path
        self.yolo = YoloService(model_path)
        self.evidence = EvidenceService(outputs_dir)
        self.outputs_dir = outputs_dir

    @staticmethod
    def _make_scene_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{ts}_{uuid.uuid4().hex[:6]}"

    @staticmethod
    def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen.")
        return img

    @staticmethod
    def _clip_xyxy(xyxy, w, h):
        x1, y1, x2, y2 = xyxy
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        return [x1, y1, x2, y2]

    def analyze_image_bytes(
        self,
        image_bytes: bytes,
        source_name: str,
        conf: float = 0.25,
        iou: float = 0.7,
        poly_points=None,  # lista [(x,y),...]
    ):
        img_bgr = self._bytes_to_bgr(image_bytes)
        h, w = img_bgr.shape[:2]

        scene_id = self._make_scene_id()
        scene_dir = os.path.join(self.outputs_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        original_path = os.path.join(scene_dir, "original.jpg")
        cv2.imwrite(original_path, img_bgr)

        # --- máscara de carretera ---
        road_mask = None
        crop_xyxy = [0, 0, w, h]

        if poly_points and len(poly_points) >= 3:
            road_mask = ROIMaskService.polygon_mask(h, w, poly_points)
            crop_xyxy = ROIMaskService.bounding_rect(poly_points)
            crop_xyxy = self._clip_xyxy(crop_xyxy, w, h)
        else:
            # si no hay polígono, analizamos todo (pero sin máscara)
            road_mask = None
            crop_xyxy = [0, 0, w, h]

        x1, y1, x2, y2 = crop_xyxy
        crop = img_bgr[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]
        if ch < 2 or cw < 2:
            raise ValueError("ROI/crop demasiado pequeña.")

        # ✅ YOLO SOLO sobre el crop
        res = self.yolo.predict(crop, conf=conf, iou=iou)

        # Remap detecciones a coords de imagen original
        detections = []
        names = res.names
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                cls_id = int(b.cls.item())
                conf_v = float(b.conf.item())
                bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0].tolist()]
                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "conf": conf_v,
                        "bbox_xyxy": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],
                    }
                )

        # Métricas usando máscara (si hay)
        metrics = MetricsService.compute(detections, w, h, road_mask=road_mask)

        # Overlay sobre imagen original
        overlay_bgr = img_bgr.copy()

        # Pintar carretera (segmento coloreado) si hay máscara
        if road_mask is not None:
            overlay_bgr = ROIMaskService.overlay_mask(overlay_bgr, road_mask, alpha=0.35, color_bgr=(0, 0, 255))
            overlay_bgr = ROIMaskService.draw_polygon_edges(overlay_bgr, poly_points, color_bgr=(0, 255, 0), thickness=3)

        # Dibujar cajas SOLO si el centro cae dentro de máscara (si existe)
        for d in detections:
            bx1, by1, bx2, by2 = [int(v) for v in d["bbox_xyxy"]]
            cx = int((bx1 + bx2) / 2)
            cy = int((by1 + by2) / 2)

            if road_mask is not None:
                if not (0 <= cx < w and 0 <= cy < h and road_mask[cy, cx] > 0):
                    continue

            cv2.rectangle(overlay_bgr, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
            cv2.putText(
                overlay_bgr,
                f"{d['class_name']} {d['conf']:.2f}",
                (bx1, max(0, by1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

        overlay_path = os.path.join(scene_dir, "overlay.jpg")
        if not cv2.imwrite(overlay_path, overlay_bgr):
            raise RuntimeError("No se pudo guardar overlay.jpg")

        result_obj = {
            "scene_id": scene_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": {"weights": os.path.basename(self.model_path), "conf": conf, "iou": iou},
            "image": {"width": w, "height": h, "source_name": source_name},
            "poly_points": poly_points,
            "crop_xyxy": crop_xyxy,
            "metrics": metrics,
            "detections": detections,
        }

        ev = self.evidence.save_evidence(scene_id, original_path, overlay_path, result_obj)

        return {
            "scene_id": scene_id,
            "metrics": metrics,
            "sha256_result_json": ev["sha256_result_json"],
            "overlay_path": os.path.abspath(overlay_path),
            "original_path": os.path.abspath(original_path),
            "result_path": os.path.abspath(ev["result_path"]),
            "scene_dir": os.path.abspath(ev["scene_dir"]),
        }

    def publish_to_bsv(self, scene_id: str, sha256_hex: str, metrics: dict) -> dict:
        wif = os.getenv("BSV_WIF", "").strip()
        if not wif:
            return {"ok": False, "error": "Falta BSV_WIF en variables de entorno."}

        network = os.getenv("BSV_NETWORK", "main").strip()

        from Model.blockchain_service import BlockchainService

        svc = BlockchainService(wif=wif, network=network)
        traffic_state = metrics.get("traffic_state") or "N/A"
        road_occ = metrics.get("road_occupancy")

        try:
            r = svc.publish_evidence(
                scene_id=scene_id,
                sha256_hex=sha256_hex,
                traffic_state=traffic_state,
                roi_occupancy=road_occ,  # reutilizamos el campo
            )
            return {"ok": True, **r}
        except Exception as ex:
            return {"ok": False, "error": str(ex)}
