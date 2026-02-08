import os
from typing import Optional, Dict, Any

from bsvlib import Wallet
from bsvlib.constants import Chain


class BlockchainService:
    """
    Publica una transacción en BSV con OP_RETURN (pushdatas)
    conteniendo: app_tag, scene_id, sha256, traffic_state, roi_occupancy
    """

    def __init__(self, wif: str, network: str = "main"):
        self.chain = Chain.MAIN if network.lower() == "main" else Chain.TEST
        self.wallet = Wallet([wif], chain=self.chain)

    def publish_evidence(
        self,
        scene_id: str,
        sha256_hex: str,
        traffic_state: str,
        roi_occupancy: Optional[float],
    ) -> Dict[str, Any]:
        # Payload corto y entendible (evita JSON largo en OP_RETURN)
        occ_str = "" if roi_occupancy is None else f"{roi_occupancy:.3f}"

        pushdatas = [
            "ROUNDABOUT",      # tag app
            scene_id,
            sha256_hex,
            traffic_state,
            occ_str,
        ]

        # Output mínimo a tu propia dirección (y el OP_RETURN va en pushdatas)
        # (bsvlib maneja el change internamente al crear/broadcast)
        addr = self.wallet.address()
        outputs = [(addr, 1)]  # 1 sat a ti mismo

        # Crea y emite tx (OP_RETURN incluido)
        # bsvlib soporta OP_RETURN con pushdatas :contentReference[oaicite:1]{index=1}
        resp = self.wallet.create_transaction(outputs=outputs, pushdatas=pushdatas).broadcast()

        # resp suele traer txid / propagated (depende versión)
        # devolvemos lo que haya de forma robusta
        txid = getattr(resp, "txid", None) or getattr(resp, "id", None) or str(resp)
        propagated = getattr(resp, "propagated", None)

        return {"txid": txid, "propagated": propagated, "raw_response": str(resp)}
