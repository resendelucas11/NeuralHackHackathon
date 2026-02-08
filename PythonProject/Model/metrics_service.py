from collections import Counter


def _bbox_center_xyxy(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _clip_box_xyxy(b, w, h):
    x1 = max(0, min(int(b[0]), w - 1))
    y1 = max(0, min(int(b[1]), h - 1))
    x2 = max(0, min(int(b[2]), w))
    y2 = max(0, min(int(b[3]), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]


class MetricsService:
    @staticmethod
    def compute(detections: list, image_w: int, image_h: int, road_mask=None):
        """
        road_mask: np.uint8 (H,W) con 0/255 (carretera definida por polígono)
        """
        # Global (por si quieres)
        counts_all = Counter([d["class_name"] for d in detections])
        total_all = sum(counts_all.values())
        density_all = total_all / float(image_w * image_h) if image_w and image_h else 0.0

        # Si no hay máscara, devolvemos globales sin estado
        if road_mask is None:
            return {
                "total_objects": total_all,
                "counts_by_class": dict(counts_all),
                "density": density_all,
                "road_occupancy": None,
                "traffic_state": None,
            }

        # Área de carretera (píxeles con mask>0)
        road_area = int((road_mask > 0).sum())
        if road_area <= 0:
            return {
                "total_objects": 0,
                "counts_by_class": {},
                "density": density_all,
                "road_occupancy": 0.0,
                "traffic_state": "FLUIDO",
            }

        # Filtrar detecciones: centro dentro de máscara
        det_in = []
        for d in detections:
            cx, cy = _bbox_center_xyxy(d["bbox_xyxy"])
            cx_i, cy_i = int(cx), int(cy)
            if 0 <= cx_i < image_w and 0 <= cy_i < image_h and road_mask[cy_i, cx_i] > 0:
                det_in.append(d)

        counts_in = Counter([d["class_name"] for d in det_in])
        total_in = sum(counts_in.values())

        # Ocupación: cuántos píxeles de carretera quedan cubiertos por cajas (aprox real)
        covered = 0
        for d in det_in:
            x1, y1, x2, y2 = _clip_box_xyxy(d["bbox_xyxy"], image_w, image_h)
            # contar píxeles de carretera dentro de la bbox
            covered += int((road_mask[y1:y2, x1:x2] > 0).sum())

        road_occupancy = covered / float(road_area)

        # Estados por ocupación (ajustables)
        if road_occupancy < 0.20:
            traffic_state = "FLUIDO"
        elif road_occupancy < 0.45:
            traffic_state = "DENSO"
        else:
            traffic_state = "ATASCO"

        return {
            "total_objects": total_in,
            "counts_by_class": dict(counts_in),
            "density": density_all,
            "road_area_pixels": road_area,
            "covered_road_pixels": covered,
            "road_occupancy": road_occupancy,
            "traffic_state": traffic_state,
        }
