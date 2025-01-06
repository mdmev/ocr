import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from typing import Optional, List

class YoloSegment:
    CLASS_COLORS = [
        sv.Color(255, 0, 0),    # Red for "Caption"
        sv.Color(0, 255, 0),    # Green for "Footnote"
        sv.Color(0, 0, 255),    # Blue for "Formula"
        sv.Color(255, 255, 0),  # Yellow for "List-item"
        sv.Color(255, 0, 255),  # Magenta for "Page-footer"
        sv.Color(0, 255, 255),  # Cyan for "Page-header"
        sv.Color(128, 0, 128),  # Purple for "Picture"
        sv.Color(128, 128, 0),  # Olive for "Section-header"
        sv.Color(128, 128, 128),# Gray for "Table"
        sv.Color(0, 128, 128),  # Teal for "Text"
        sv.Color(128, 0, 0)     # Maroon for "Title"
    ]
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(self.model_path)

    def run_inference(self, image) -> sv.Detections:
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    def filter_text_detections(self, detections: sv.Detections, exclude_classes: Optional[List[int]] = None) -> sv.Detections:
        if not exclude_classes:
            return detections
        mask_not_excluded = ~np.isin(detections.class_id, exclude_classes)
        filtered_detections = detections[mask_not_excluded]
        return filtered_detections

    def annotate_image(self, image, detections: sv.Detections) -> None:

        box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette(self.CLASS_COLORS),
            thickness=3
        )
        label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette(self.CLASS_COLORS),
            text_color=sv.Color(255, 255, 255)
        )
        annotated_image = box_annotator.annotate(
            scene=image.copy(),
            detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image,
            detections=detections
        )
        return annotated_image
