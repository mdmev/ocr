import numpy as np
from typing import Tuple, Dict, List

class BoundingBoxMatcher:
    """
    A class to match bounding boxes between two images based on spatial proximity,
    class consistency, and a configurable distance threshold.
    
    Attributes:
    -----------
    distance_threshold : float
        The maximum Euclidean distance between two box centers
        for a valid match.
    """

    def __init__(self, distance_threshold: float = 50.0):
        """
        Initialize the matcher with a chosen distance threshold.
        """
        self.distance_threshold = distance_threshold

    @staticmethod
    def _box_center(xyxy: np.ndarray) -> Tuple[float, float]:
        """
        Compute the (x_center, y_center) of a bounding box in xyxy format.
        xyxy is [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return cx, cy

    @staticmethod
    def _euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
        """
        Compute the Euclidean distance between two points (x1, y1) and (x2, y2).
        """
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def match(self,
              left_detections,
              right_detections) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Match bounding boxes between two sets of detections (left and right),
        ensuring:
          - Only boxes of the same class_name are paired.
          - Euclidean distance between centers <= distance_threshold.
          - Each box in the right image is matched at most once.

        Parameters:
        -----------
        left_detections : sv.Detections
            Detections object for the left image.
        right_detections : sv.Detections
            Detections object for the right image.

        Returns:
        --------
        matches_by_class : Dict[str, List[Tuple[int, int, float]]]
            A dictionary keyed by class_name, where each value is a list
            of tuples (left_idx, right_idx, distance).
        """
        # Group box indices by class for left and right
        left_by_class = {}
        right_by_class = {}

        for i, cls_name in enumerate(left_detections.data['class_name']):
            left_by_class.setdefault(cls_name, []).append(i)

        for j, cls_name in enumerate(right_detections.data['class_name']):
            right_by_class.setdefault(cls_name, []).append(j)

        # Prepare result dictionary
        matches_by_class = {}

        # Iterate over each class present in left detections
        for cls_name, left_indices in left_by_class.items():
            if cls_name not in right_by_class:
                # No corresponding class in right detections
                continue

            right_indices = right_by_class[cls_name]
            matched_pairs = []
            matched_right_indices = set()

            # For each left bounding box, find the closest right bounding box
            for left_idx in left_indices:
                left_box = left_detections.xyxy[left_idx]
                left_center = self._box_center(left_box)

                best_match = None
                min_dist = float('inf')

                for right_idx in right_indices:
                    # Skip if already matched
                    if right_idx in matched_right_indices:
                        continue

                    right_box = right_detections.xyxy[right_idx]
                    right_center = self._box_center(right_box)
                    dist = self._euclidean_distance(left_center, right_center)

                    # Track the minimal distance for this left box
                    if dist < min_dist:
                        min_dist = dist
                        best_match = right_idx

                # If the best match is within the acceptable threshold, record it
                if best_match is not None and min_dist <= self.distance_threshold:
                    matched_pairs.append((left_idx, best_match, min_dist))
                    matched_right_indices.add(best_match)

            matches_by_class[cls_name] = matched_pairs

        return matches_by_class
