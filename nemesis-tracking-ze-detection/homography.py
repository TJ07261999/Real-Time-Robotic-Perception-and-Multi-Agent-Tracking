from typing import Tuple

import cv2
import numpy as np

from config import (
    CALIBRATION_CLICKS,
    CALIBRATION_SQUARE_SIZE_FT,
    CALIBRATION_TILE_X,
    CALIBRATION_TILE_Y,
    TILE_SIZE_FT,
)


def compute_homography() -> np.ndarray:
    pixel_pts = np.array(CALIBRATION_CLICKS, dtype=np.float32)

    cal_x = CALIBRATION_TILE_X * TILE_SIZE_FT
    cal_y = CALIBRATION_TILE_Y * TILE_SIZE_FT

    arena_pts = np.array(
        [
            [cal_x, cal_y],
            [cal_x, cal_y + CALIBRATION_SQUARE_SIZE_FT],
            [cal_x + CALIBRATION_SQUARE_SIZE_FT, cal_y + CALIBRATION_SQUARE_SIZE_FT],
            [cal_x + CALIBRATION_SQUARE_SIZE_FT, cal_y],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(pixel_pts, arena_pts)
    return H


def pixel_to_arena(px: float, py: float, H: np.ndarray) -> Tuple[float, float]:
    pt = np.array([[[px, py]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return float(transformed[0, 0, 0]), float(transformed[0, 0, 1])


def arena_to_pixel(ax: float, ay: float, H_inv: np.ndarray) -> Tuple[int, int]:
    pt = np.array([[[ax, ay]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H_inv)
    return int(transformed[0, 0, 0]), int(transformed[0, 0, 1])


class CoordinateTransformer:
    def __init__(self):
        self.H = compute_homography()
        self.H_inv = np.linalg.inv(self.H)

    def pixel_to_arena(self, px: float, py: float) -> Tuple[float, float]:
        return pixel_to_arena(px, py, self.H)

    def arena_to_pixel(self, ax: float, ay: float) -> Tuple[int, int]:
        return arena_to_pixel(ax, ay, self.H_inv)
