from typing import Callable, Tuple

import cv2
import numpy as np

from config import (
    ARENA_HEIGHT_FT,
    ARENA_WIDTH_FT,
    CELL_SIZE_FT,
    COLORS,
    GRID_CELLS_PER_SIDE,
)
from homography import arena_to_pixel


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)


def draw_text_styled(
    img, text, pos, font_scale=0.5, color=(255, 255, 255), thickness=1, shadow=True
):
    if shadow:
        cv2.putText(
            img,
            str(text),
            (pos[0] + 1, pos[1] + 1),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        str(text),
        pos,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# COORDINATE HELPERS


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def create_arena_to_sim_transform(width: int, height: int) -> Callable:
    scale = min(width, height) / ARENA_WIDTH_FT * 0.9
    offset_x = (width - ARENA_WIDTH_FT * scale) / 2
    offset_y = (height - ARENA_HEIGHT_FT * scale) / 2

    def transform(ax: float, ay: float) -> Tuple[int, int]:
        return int(offset_x + ax * scale), int(
            offset_y + (ARENA_HEIGHT_FT - ay) * scale
        )

    return transform


# GRID AND HAZARD DRAWING


def draw_grid_sim(img, arena_to_sim: Callable, major_interval: int = 4):
    grid_color = (210, 210, 210)
    major_color = (180, 180, 180)

    for i in range(1, GRID_CELLS_PER_SIDE):
        ft = i * CELL_SIZE_FT
        is_major = i % major_interval == 0
        color = major_color if is_major else grid_color

        p1, p2 = arena_to_sim(ft, 0), arena_to_sim(ft, ARENA_HEIGHT_FT)
        cv2.line(img, p1, p2, color, 1)

        p1, p2 = arena_to_sim(0, ft), arena_to_sim(ARENA_WIDTH_FT, ft)
        cv2.line(img, p1, p2, color, 1)


def draw_hazard_cells_sim(img, hazard_grid, arena_to_sim: Callable):
    hazard_color = COLORS.get("HAZARD", (40, 40, 80))

    for cy in range(GRID_CELLS_PER_SIDE):
        for cx in range(GRID_CELLS_PER_SIDE):
            if hazard_grid[cy, cx]:
                ax1, ay1 = cx * CELL_SIZE_FT, cy * CELL_SIZE_FT
                ax2, ay2 = ax1 + CELL_SIZE_FT, ay1 + CELL_SIZE_FT

                p1 = arena_to_sim(ax1, ay1)
                p2 = arena_to_sim(ax2, ay1)
                p3 = arena_to_sim(ax2, ay2)
                p4 = arena_to_sim(ax1, ay2)

                pts = np.array([p1, p2, p3, p4], dtype=np.int32)
                cv2.fillPoly(img, [pts], hazard_color)


def draw_grid_video(img, H_inv, grid_color=(100, 100, 100)):
    for i in range(1, GRID_CELLS_PER_SIDE):
        ft = i * CELL_SIZE_FT
        p1 = arena_to_pixel(ft, 0, H_inv)
        p2 = arena_to_pixel(ft, ARENA_HEIGHT_FT, H_inv)
        cv2.line(img, p1, p2, grid_color, 1)

        p1 = arena_to_pixel(0, ft, H_inv)
        p2 = arena_to_pixel(ARENA_WIDTH_FT, ft, H_inv)
        cv2.line(img, p1, p2, grid_color, 1)


def draw_hazard_cells_video(img, hazard_grid, H_inv, color=(0, 0, 180), alpha=0.4):
    overlay = img.copy()

    for cy in range(GRID_CELLS_PER_SIDE):
        for cx in range(GRID_CELLS_PER_SIDE):
            if hazard_grid[cy, cx]:
                ax, ay = cx * CELL_SIZE_FT, cy * CELL_SIZE_FT
                corners = [
                    arena_to_pixel(ax, ay, H_inv),
                    arena_to_pixel(ax + CELL_SIZE_FT, ay, H_inv),
                    arena_to_pixel(ax + CELL_SIZE_FT, ay + CELL_SIZE_FT, H_inv),
                    arena_to_pixel(ax, ay + CELL_SIZE_FT, H_inv),
                ]
                cv2.fillPoly(overlay, [np.array(corners, dtype=np.int32)], color)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# ROBOT DRAWING


def draw_velocity_arrow(
    img,
    start: Tuple[int, int],
    velocity: Tuple[float, float],
    color: Tuple[int, int, int],
    arrow_length: int = 30,
    min_speed: float = 0.5,
):
    vx, vy = velocity
    speed = np.sqrt(vx * vx + vy * vy)

    if speed < min_speed:
        return

    end_x = int(start[0] + (vx / speed) * arrow_length)
    end_y = int(start[1] - (vy / speed) * arrow_length)  # Flip Y
    cv2.arrowedLine(img, start, (end_x, end_y), color, 2, tipLength=0.3)


def draw_distance_line(img, p1: Tuple[int, int], p2: Tuple[int, int], distance: float):
    if distance < 3:
        color = (0, 0, 255)  # Red
    elif distance < 6:
        color = (0, 165, 255)  # Orange
    else:
        color = (0, 200, 0)  # Green

    cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)


def get_distance_color(distance: float) -> Tuple[int, int, int]:
    if distance < 3.0:
        return (0, 0, 255)
    elif distance < 6.0:
        return (0, 165, 255)
    else:
        return (0, 255, 0)
