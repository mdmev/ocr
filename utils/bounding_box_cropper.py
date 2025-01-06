import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

class BoundingBoxCropper:
    """
    Crops bounding boxes from left and right images based on matched detections
    and plots them side by side.
    """

    def __init__(self):
        pass

    def crop_boxes_and_display(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        matches_by_class: Dict[str, List[Tuple[int, int, float]]],
        left_detections,
        right_detections
    ) -> List[Tuple[np.ndarray, np.ndarray, str, float]]:
        """
        For each matched bounding box pair, crops the region from both
        left and right images and displays them side by side.
        
        Parameters:
        -----------
        left_image : np.ndarray
            The left image as a NumPy BGR array.
        right_image : np.ndarray
            The right image as a NumPy BGR array.
        matches_by_class : Dict[str, List[Tuple[int, int, float]]]
            The dictionary of matched bounding boxes, keyed by class_name:
            {
              "Text": [(left_idx, right_idx, distance), ...],
              "Picture": [...],
              ...
            }
        left_detections : sv.Detections
            Detections for the left image (containing .xyxy and .class_name).
        right_detections : sv.Detections
            Detections for the right image (containing .xyxy and .class_name).
        """
        
        boxes = []
        for cls_name, matches in matches_by_class.items():
            for (l_idx, r_idx, dist) in matches:
                
                left_box = left_detections.xyxy[l_idx]   # [x1, y1, x2, y2]
                right_box = right_detections.xyxy[r_idx] # [x1, y1, x2, y2]

                lx1, ly1, lx2, ly2 = map(int, left_box)
                rx1, ry1, rx2, ry2 = map(int, right_box)

                left_crop = left_image[ly1:ly2, lx1:lx2]

                right_crop = right_image[ry1:ry2, rx1:rx2]
                
                boxes.append((left_crop, right_crop, cls_name, dist))
        return boxes

