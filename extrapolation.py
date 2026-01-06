import cv2
import numpy as np

class SprayConeBackfill:
    def __init__(self, origin, min_points=50):
        self.origin = np.array(origin, dtype=np.float32)
        self.min_points = min_points

    def backfill(self, detected_mask):
        h, w = detected_mask.shape
        ox, oy = self.origin

        # Only consider right half for detection
        xs, ys = np.where(detected_mask[:, w//2:] > 0)
        if len(xs) < self.min_points:
            return np.zeros_like(detected_mask)

        # Shift xs back to full image coordinates
        xs = xs + w // 2
        points = np.column_stack((xs, ys))

        # Find closest detected x to origin
        first_x = np.min(xs)

        # Distance from origin to first detection
        D = first_x - ox
        if D <= 5:
            return np.zeros_like(detected_mask)

        # Estimate width at first detection
        near_mask = np.abs(xs - first_x) < 3
        width = np.ptp(ys[near_mask])

        if width < 2:
            return np.zeros_like(detected_mask)

        half_width = width / 2.0
        theta = np.arctan2(half_width, D)

        # Create backfill mask
        backfill = np.zeros_like(detected_mask)
        center_y = oy

        for x in range(int(ox), int(first_x)):
            dx = x - ox
            if dx <= 0:
                continue

            curr_half_width = np.tan(theta) * dx
            y1 = int(center_y - curr_half_width)
            y2 = int(center_y + curr_half_width)

            if y2 < 0 or y1 >= h:
                continue

            y1 = max(0, y1)
            y2 = min(h - 1, y2)

            backfill[y1:y2, x] = 255

        return backfill
