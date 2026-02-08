import os
import sys
import json
import base64
import subprocess

import flet as ft
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

from Controller.app_controller import AppController

MODEL_PATH = os.path.join("Yolo", "best_roundabout.pt")
ROI_PICKER_PATH = os.path.join("View", "roi_picker.py")


def pick_file_dialog():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png")],
    )
    root.destroy()
    return path if path else None


def load_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("No se pudo abrir la imagen")
    return img


def img_to_data_uri(img_bgr):
    ok, buf = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def draw_polygon_overlay(img_bgr: np.ndarray, points_xy):
    out = img_bgr.copy()

    if points_xy and len(points_xy) >= 3:
        mask = np.zeros(out.shape[:2], dtype=np.uint8)
        pts = np.array(points_xy, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

        alpha = 0.35
        red = np.zeros_like(out, dtype=np.uint8)
        red[:] = (0, 0, 255)
        m = mask > 0
        out[m] = (out[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)

        cv2.polylines(out, [pts], True, (0, 255, 0), 3)

    return out


def run_roi_picker(image_path: str):
    """
    Ejecuta OpenCV ROI picker en OTRO proceso para que el ratón funcione siempre.
    Devuelve lista de puntos [(x,y),...] o None (cancelado).
    """
    cmd = [sys.executable, ROI_PICKER_PATH, image_path]
    p = subprocess.run(cmd, capture_output=True, text=True)

    # si no imprime nada
    stdout = (p.stdout or "").strip()
    if not stdout:
        return None

    try:
        data = json.loads(stdout)
    except Exception:
        return None

    if data.get("ok"):
        pts = data.get("points")
        return [tuple(map(int, xy)) for xy in pts]
    return None


def main(page: ft.Page):
    page.title = "Roundabout Analyzer – YOLO + Segmento"
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO

    controller = AppController(MODEL_PATH, outputs_dir="outputs")

    selected_path = None
    poly_points = None

    status = ft.Text("")
    traffic_text = ft.Text("Estado tráfico: -", size=22, weight="bold")

    # TU Flet obliga a pasar src en Image:
    overlay_img = ft.Image(src="", width=900)

    metrics_box = ft.Column()

    conf_slider = ft.Slider(min=0.05, max=0.95, value=0.25)
    iou_slider = ft.Slider(min=0.05, max=0.95, value=0.70)

    def on_pick(_):
        nonlocal selected_path, poly_points
        p = pick_file_dialog()
        if not p:
            status.value = "Selección cancelada"
            page.update()
            return

        selected_path = p
        poly_points = None

        img = load_bgr(p)
        overlay_img.src = img_to_data_uri(img)

        status.value = f"Imagen cargada: {os.path.basename(p)}"
        page.update()

    def on_define_poly(_):
        nonlocal poly_points
        if not selected_path:
            status.value = "Primero selecciona una imagen"
            page.update()
            return

        if not os.path.exists(ROI_PICKER_PATH):
            status.value = f"❌ No existe {ROI_PICKER_PATH}. Crea el archivo roi_picker.py"
            page.update()
            return

        status.value = "🖱️ Se abrirá OpenCV (otra ventana). Dibuja el polígono y pulsa ENTER."
        page.update()

        pts = run_roi_picker(selected_path)
        if pts and len(pts) >= 3:
            poly_points = pts

            img = load_bgr(selected_path)
            preview = draw_polygon_overlay(img, poly_points)
            overlay_img.src = img_to_data_uri(preview)

            status.value = f"✅ Segmento definido ({len(poly_points)} puntos)"
        else:
            poly_points = None
            status.value = "ℹ️ Segmento cancelado"

        page.update()

    def on_analyze(_):
        nonlocal poly_points
        if not selected_path:
            status.value = "Selecciona una imagen primero"
            page.update()
            return

        status.value = "Analizando..."
        page.update()

        with open(selected_path, "rb") as f:
            img_bytes = f.read()

        out = controller.analyze_image_bytes(
            image_bytes=img_bytes,
            source_name=os.path.basename(selected_path),
            conf=float(conf_slider.value),
            iou=float(iou_slider.value),
            poly_points=poly_points,
        )

        overlay = load_bgr(out["overlay_path"])
        overlay_img.src = img_to_data_uri(overlay)

        m = out["metrics"]
        traffic_text.value = f"Estado tráfico: {m.get('traffic_state', '-')}"
        metrics_box.controls.clear()
        metrics_box.controls.append(ft.Text(f"Vehículos (en segmento): {m.get('total_objects', 0)}"))
        if m.get("road_occupancy") is not None:
            metrics_box.controls.append(ft.Text(f"Ocupación carretera: {m['road_occupancy']:.2f}"))

        status.value = "✅ Análisis completado"
        page.update()

    page.add(
        ft.Text("🚦 Roundabout Analyzer", size=28, weight="bold"),
        ft.Row([
            ft.ElevatedButton("📂 Imagen", on_click=on_pick),
            ft.ElevatedButton("🟥 Definir segmento", on_click=on_define_poly),
            ft.ElevatedButton("🔍 Analizar", on_click=on_analyze),
        ]),
        ft.Text("Confianza"),
        conf_slider,
        ft.Text("IoU"),
        iou_slider,
        status,
        traffic_text,
        overlay_img,
        metrics_box,
    )


if __name__ == "__main__":
    ft.run(main)
