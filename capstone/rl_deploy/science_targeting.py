"""
Science Targeting - Rock/Boulder Target Selection
=================================================

This module finds small-rock or big-rock targets from a segmentation mask,
chooses the best candidate, and converts its image position into gimbal
pan/tilt angles.

It does NOT move the rover.
It does NOT control the gimbal directly.
It only answers: "Is there a good rock target, and where should the gimbal point?"
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import numpy as np


@dataclass
class RockTarget:
    """Information about one selected rock/boulder target."""
    class_id: int
    class_name: str
    center_px: Tuple[int, int]
    area_px: int
    area_frac: float
    distance_m: float
    pan_deg: float
    tilt_deg: float
    score: float


class RockTargetSelector:
    """
    Selects a science target from a semantic segmentation mask.

    Expected segmentation labels (current 5-class TerrainSegmenter output -
    this used to be written for an older 4-class model where 2/3 were the
    rock classes; that model's mask now has 0=ground, 1=big rock,
    2=small rock, 3=crater, 4=sky, so the rock classes moved to 1/2):
        0 = ground
        1 = big rock
        2 = small rock
        3 = crater
        4 = sky
    """

    CLASS_NAMES = {
        1: "big_rock",
        2: "small_rock",
    }

    def __init__(
        self,
        preferred_class: str = "big",
        min_distance_m: float = 0.5,
        max_distance_m: float = 2.5,
        min_area_frac: float = 0.002,
        horizontal_fov_deg: float = 73.0,
        vertical_fov_deg: float = 58.0,
        laser_pan_offset_deg: float = 0.0,
        laser_tilt_offset_deg: float = 0.0,
    ):
        self.preferred_class = preferred_class
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.min_area_frac = min_area_frac
        self.hfov_rad = math.radians(horizontal_fov_deg)
        self.vfov_rad = math.radians(vertical_fov_deg)
        self.laser_pan_offset_deg = laser_pan_offset_deg
        self.laser_tilt_offset_deg = laser_tilt_offset_deg

    def find_target(
        self,
        seg_mask: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
    ) -> Optional[RockTarget]:
        """
        Find the best rock target.

        seg_mask:
            2D array where pixels are class labels.

        depth_frame:
            Optional 2D depth array in meters. If given, the selector estimates
            target distance by sampling a small patch at the rock center.

        Returns:
            RockTarget if a good target exists, otherwise None.
        """
        if seg_mask is None or seg_mask.ndim != 2:
            return None

        h, w = seg_mask.shape
        frame_area = h * w
        min_area_px = int(frame_area * self.min_area_frac)

        best = None
        best_score = -1.0

        for class_id in self._candidate_classes():
            binary = (seg_mask == class_id).astype(np.uint8)
            components = self._connected_components(binary)

            for area_px, cx, cy in components:
                if area_px < min_area_px:
                    continue

                cx_i, cy_i = int(round(cx)), int(round(cy))

                # Avoid targets on the very edge of the image.
                if cx_i < 5 or cx_i >= w - 5 or cy_i < 5 or cy_i >= h - 5:
                    continue

                distance_m = self._estimate_depth(
                    depth_frame=depth_frame,
                    cx=cx_i,
                    cy=cy_i,
                    mask_shape=(h, w),
                )

                if distance_m is not None:
                    if distance_m < self.min_distance_m or distance_m > self.max_distance_m:
                        continue
                else:
                    # If no depth is available, keep candidate but mark unknown distance.
                    distance_m = float("nan")

                pan_deg, tilt_deg = self._pixel_to_gimbal_angles(cx_i, cy_i, w, h)
                area_frac = area_px / frame_area

                score = self._score_candidate(
                    class_id=class_id,
                    area_frac=area_frac,
                    cx=cx_i,
                    cy=cy_i,
                    w=w,
                    h=h,
                    distance_m=distance_m,
                )

                if score > best_score:
                    best_score = score
                    best = RockTarget(
                        class_id=class_id,
                        class_name=self.CLASS_NAMES[class_id],
                        center_px=(cx_i, cy_i),
                        area_px=area_px,
                        area_frac=area_frac,
                        distance_m=distance_m,
                        pan_deg=pan_deg,
                        tilt_deg=tilt_deg,
                        score=score,
                    )

        return best

    def _candidate_classes(self):
        if self.preferred_class == "small":
            return [2, 1]
        return [1, 2]

    def _connected_components(self, binary: np.ndarray):
        """
        Find connected blobs in a binary image using only numpy/Python.

        Returns:
            list of (area_px, centroid_x, centroid_y)
        """
        h, w = binary.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []

        for y in range(h):
            for x in range(w):
                if binary[y, x] == 0 or visited[y, x]:
                    continue

                stack = [(x, y)]
                visited[y, x] = True

                area = 0
                sum_x = 0
                sum_y = 0

                while stack:
                    px, py = stack.pop()
                    area += 1
                    sum_x += px
                    sum_y += py

                    for ny in range(py - 1, py + 2):
                        for nx in range(px - 1, px + 2):
                            if nx == px and ny == py:
                                continue
                            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                                continue
                            if visited[ny, nx]:
                                continue
                            if binary[ny, nx] == 0:
                                continue

                            visited[ny, nx] = True
                            stack.append((nx, ny))

                cx = sum_x / area
                cy = sum_y / area
                components.append((area, cx, cy))

        return components

    def _estimate_depth(
        self,
        depth_frame: Optional[np.ndarray],
        cx: int,
        cy: int,
        mask_shape: Tuple[int, int],
        patch_radius: int = 3,
    ) -> Optional[float]:
        """
        Estimate target distance by mapping segmentation-mask coordinates
        into the depth-frame coordinates and taking the median depth nearby.
        """
        if depth_frame is None or depth_frame.ndim != 2:
            return None

        mask_h, mask_w = mask_shape
        depth_h, depth_w = depth_frame.shape

        dx = int(round(cx * depth_w / mask_w))
        dy = int(round(cy * depth_h / mask_h))

        dx = max(patch_radius, min(depth_w - patch_radius - 1, dx))
        dy = max(patch_radius, min(depth_h - patch_radius - 1, dy))

        patch = depth_frame[
            dy - patch_radius: dy + patch_radius + 1,
            dx - patch_radius: dx + patch_radius + 1,
        ]

        valid = patch[np.isfinite(patch) & (patch > 0.02)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    def _pixel_to_gimbal_angles(self, cx: int, cy: int, w: int, h: int):
        """
        Convert a pixel center into gimbal pan/tilt angles.

        Positive pan means target is to the right of image center.
        Positive tilt means target is above image center.
        """
        center_x_norm = cx / w
        center_y_norm = cy / h

        pan_rad = (center_x_norm - 0.5) * self.hfov_rad
        tilt_rad = -(center_y_norm - 0.5) * self.vfov_rad

        pan_deg = math.degrees(pan_rad) + self.laser_pan_offset_deg
        tilt_deg = math.degrees(tilt_rad) + self.laser_tilt_offset_deg

        return pan_deg, tilt_deg

    def _score_candidate(
        self,
        class_id: int,
        area_frac: float,
        cx: int,
        cy: int,
        w: int,
        h: int,
        distance_m: float,
    ) -> float:
        """
        Higher score means better science target.
        """
        class_score = 1.0
        if self.preferred_class == "big" and class_id == 1:
            class_score = 1.5
        elif self.preferred_class == "small" and class_id == 2:
            class_score = 1.5

        size_score = min(area_frac / 0.04, 1.0)

        nx = (cx / w) - 0.5
        ny = (cy / h) - 0.5
        center_dist = math.sqrt(nx * nx + ny * ny)
        center_score = max(0.0, 1.0 - center_dist * 2.0)

        if math.isnan(distance_m):
            distance_score = 0.5
        else:
            ideal = (self.min_distance_m + self.max_distance_m) / 2.0
            half_range = max((self.max_distance_m - self.min_distance_m) / 2.0, 1e-6)
            distance_score = max(0.0, 1.0 - abs(distance_m - ideal) / half_range)

        return (
            0.40 * class_score
            + 0.25 * size_score
            + 0.25 * center_score
            + 0.10 * distance_score
        )
