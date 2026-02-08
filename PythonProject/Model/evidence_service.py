import os
import json
import hashlib
from datetime import datetime, timezone

class EvidenceService:
    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    @staticmethod
    def _sha256_bytes(data: bytes) -> str:
        h = hashlib.sha256()
        h.update(data)
        return h.hexdigest()

    @staticmethod
    def _canonical_json_bytes(obj: dict) -> bytes:
        # JSON canónico (claves ordenadas) => hash estable
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def save_evidence(self, scene_id: str, original_path: str, overlay_path: str, result_obj: dict):
        scene_dir = os.path.join(self.base_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        result_path = os.path.join(scene_dir, "result.json")
        with open(result_path, "wb") as f:
            payload = self._canonical_json_bytes(result_obj)
            f.write(payload)

        sha256 = self._sha256_bytes(payload)
        with open(os.path.join(scene_dir, "sha256.txt"), "w", encoding="utf-8") as f:
            f.write(sha256)

        # Guarda punteros a archivos (ya los has escrito en controller)
        return {
            "scene_id": scene_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sha256_result_json": sha256,
            "scene_dir": scene_dir,
            "original_path": original_path,
            "overlay_path": overlay_path,
            "result_path": result_path
        }
