"""RF-DETR person detector."""
from __future__ import annotations

import logging
import warnings

import numpy as np

VARIANTS = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
            "base": "RFDETRBase", "large": "RFDETRLarge"}
COCO_PERSON = 1


class PersonDetector:
    """Returns `(boxes (N, 4) x1 y1 x2 y2, scores (N,))` for the people in an RGB frame, best first."""

    def __init__(self, variant: str = "medium", confidence: float = 0.5, max_persons: int = 4):
        import rfdetr

        logging.getLogger("rf-detr").setLevel(logging.ERROR)
        if variant not in VARIANTS:
            raise ValueError(f"unknown RF-DETR variant {variant!r}: choose one of {', '.join(VARIANTS)}")
        self.model = getattr(rfdetr, VARIANTS[variant])()
        if hasattr(self.model, "optimize_for_inference"):   # rfdetr < 1.10
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.optimize_for_inference()
        self.confidence, self.max_persons = confidence, max_persons

    def __call__(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from PIL import Image

        det = self.model.predict(Image.fromarray(image_rgb), threshold=self.confidence)
        keep = det.class_id == COCO_PERSON if det.class_id is not None else np.zeros(len(det), bool)
        boxes, scores = det.xyxy[keep].astype(np.float32), det.confidence[keep].astype(np.float32)
        order = np.argsort(-scores)[: self.max_persons]
        return boxes[order], scores[order]
