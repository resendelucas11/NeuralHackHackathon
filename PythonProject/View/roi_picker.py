import sys
import json
import cv2
import numpy as np


def pick_polygon_roi_opencv(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return None

    points = []
    base = img.copy()
    window = "ROI Poligono | Click izq=añadir | Click der=borrar | U=undo | C=clear | ENTER=OK | ESC=Cancelar"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    def redraw():
        temp = base.copy()

        if len(points) >= 3:
            mask = np.zeros(temp.shape[:2], dtype=np.uint8)
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)

            alpha = 0.35
            red = np.zeros_like(temp)
            red[:] = (0, 0, 255)
            temp[mask > 0] = (temp[mask > 0] * (1 - alpha) + red[mask > 0] * alpha).astype(np.uint8)

            cv2.polylines(temp, [pts], True, (0, 255, 0), 3)

        for p in points:
            cv2.circle(temp, p, 6, (0, 255, 0), -1)

        for i in range(len(points) - 1):
            cv2.line(temp, points[i], points[i + 1], (0, 255, 0), 2)

        cv2.putText(
            temp,
            f"Puntos: {len(points)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow(window, temp)

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
                redraw()

    cv2.setMouseCallback(window, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None

        if key in (ord("u"), ord("U")):
            if points:
                points.pop()
                redraw()

        if key in (ord("c"), ord("C")):
            points = []
            redraw()

        if key in (13, 10):  # ENTER
            cv2.destroyAllWindows()
            if len(points) >= 3:
                return points
            return None


if __name__ == "__main__":
    # Usage: python roi_picker.py <image_path>
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "missing_image_path"}))
        sys.exit(2)

    image_path = sys.argv[1]
    pts = pick_polygon_roi_opencv(image_path)

    if pts is None:
        print(json.dumps({"ok": False, "cancelled": True}))
        sys.exit(0)

    print(json.dumps({"ok": True, "points": pts}))
    sys.exit(0)
