from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    ARENA_HEIGHT_FT,
    ARENA_WIDTH_FT,
    CELL_SIZE_FT,
    CLOSE_PROXIMITY_FT,
    COLORS,
    GRID_CELLS_PER_SIDE,
    MAX_ASPECT_RATIO,
    MAX_BOX_AREA_PX,
    MAX_ROBOT_SPEED_FT_PER_SEC,
    MIN_ASPECT_RATIO,
    MIN_BOX_AREA_PX,
    MIN_CONFIDENCE,
)
from homography import CoordinateTransformer, pixel_to_arena


def create_kalman_filter(initial_x: float, initial_y: float) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
    kf.statePre = np.array([[initial_x], [initial_y], [0], [0]], dtype=np.float32)
    kf.statePost = np.array([[initial_x], [initial_y], [0], [0]], dtype=np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


class Track:

    def __init__(
        self, track_id: int, color: Tuple[int, int, int], kf: cv2.KalmanFilter
    ):
        self.id = track_id
        self.color = color
        self.kf = kf
        self.age = 0

    @property
    def position(self) -> Tuple[float, float]:
        state = self.kf.statePost
        return float(state[0, 0]), float(state[1, 0])


# Tracking parameters
NORMAL_ASSIGNMENT_DIST_FT = 12.0
CONTACT_ASSIGNMENT_DIST_FT = 3.5
CLOSE_ROBOT_DIST_FT = 6.0
PHYSICS_REJECTION_MULTIPLIER = 2.5
MERGED_BOX_SIZE_RATIO = 1.5
MIN_SPLIT_DISTANCE_FT = 1.0
VELOCITY_SMOOTHING_ALPHA = 0.12
VELOCITY_HISTORY_SIZE = 12
MIN_DISPLAY_SPEED = 0.8
POSITION_ALPHA = 0.5


class RobotTracker:

    def __init__(self, auto_init: bool = True):
        self.transformer = CoordinateTransformer()
        self.H = self.transformer.H
        self.H_inv = self.transformer.H_inv

        self.fps = 30.0
        self.dt = 1.0 / 30.0
        self.max_movement_per_frame = (
            MAX_ROBOT_SPEED_FT_PER_SEC * self.dt * PHYSICS_REJECTION_MULTIPLIER
        )

        self.labels = ["USC", "ENEMY"]
        self.colors = [COLORS["USC"], COLORS["ENEMY"]]

        self.tracks: List[Optional[Track]] = [None, None]
        self.auto_init = auto_init
        self.initialized = False

        self.last_detection_pos: List[Optional[Tuple[float, float]]] = [None, None]
        self.smoothed_pos: List[Optional[Tuple[float, float]]] = [None, None]

        self.raw_velocity: List[Tuple[float, float]] = [(0, 0), (0, 0)]
        self.smoothed_velocity: List[Tuple[float, float]] = [(0, 0), (0, 0)]
        self.velocity_history: List[deque] = [
            deque(maxlen=VELOCITY_HISTORY_SIZE),
            deque(maxlen=VELOCITY_HISTORY_SIZE),
        ]
        self.display_velocity: List[Tuple[float, float]] = [(0, 0), (0, 0)]
        self.speed_ft_per_sec: List[float] = [0.0, 0.0]

        self.frames_since_detection: List[int] = [0, 0]
        self.typical_box_size: List[Optional[float]] = [None, None]

        self.frame_count = 0
        self.hazard_grid = np.zeros(
            (GRID_CELLS_PER_SIDE, GRID_CELLS_PER_SIDE), dtype=bool
        )

        self._in_contact = False
        self._last_robot_distance = float("inf")

    def set_fps(self, fps: float):
        self.fps = max(fps, 1.0)
        self.dt = 1.0 / self.fps
        self.max_movement_per_frame = (
            MAX_ROBOT_SPEED_FT_PER_SEC * self.dt * PHYSICS_REJECTION_MULTIPLIER
        )

    def set_dt(self, dt: float):
        if dt > 0.001:
            self.dt = min(dt, 0.5)
            self.fps = 1.0 / self.dt
            self.max_movement_per_frame = (
                MAX_ROBOT_SPEED_FT_PER_SEC * self.dt * PHYSICS_REJECTION_MULTIPLIER
            )

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _get_robot_distance(self) -> float:
        if self.smoothed_pos[0] is None or self.smoothed_pos[1] is None:
            return float("inf")
        return self._distance(self.smoothed_pos[0], self.smoothed_pos[1])

    def _get_average_box_size(self) -> float:
        sizes = [s for s in self.typical_box_size if s is not None]
        return sum(sizes) / len(sizes) if sizes else 3.0

    def _is_merged_box(self, box_size: float) -> bool:
        return box_size > self._get_average_box_size() * MERGED_BOX_SIZE_RATIO

    def _create_track(self, slot: int, ax: float, ay: float) -> Track:
        kf = create_kalman_filter(ax, ay)
        return Track(slot, self.colors[slot], kf)

    def _update_velocity(self, idx: int, new_pos: Tuple[float, float]):
        if self.last_detection_pos[idx] is None:
            return

        old = self.last_detection_pos[idx]
        raw_vx = (new_pos[0] - old[0]) / self.dt if self.dt > 0 else 0
        raw_vy = (new_pos[1] - old[1]) / self.dt if self.dt > 0 else 0

        raw_speed = np.sqrt(raw_vx**2 + raw_vy**2)
        if raw_speed > MAX_ROBOT_SPEED_FT_PER_SEC:
            scale = MAX_ROBOT_SPEED_FT_PER_SEC / raw_speed
            raw_vx *= scale
            raw_vy *= scale

        self.raw_velocity[idx] = (raw_vx, raw_vy)
        self.velocity_history[idx].append((raw_vx, raw_vy))

        old_vx, old_vy = self.smoothed_velocity[idx]
        self.smoothed_velocity[idx] = (
            VELOCITY_SMOOTHING_ALPHA * raw_vx + (1 - VELOCITY_SMOOTHING_ALPHA) * old_vx,
            VELOCITY_SMOOTHING_ALPHA * raw_vy + (1 - VELOCITY_SMOOTHING_ALPHA) * old_vy,
        )

        if len(self.velocity_history[idx]) >= 3:
            hist = list(self.velocity_history[idx])
            avg_vx = sum(v[0] for v in hist) / len(hist)
            avg_vy = sum(v[1] for v in hist) / len(hist)
            speed = np.sqrt(avg_vx**2 + avg_vy**2)

            if speed > MIN_DISPLAY_SPEED:
                self.display_velocity[idx] = (avg_vx, avg_vy)
                self.speed_ft_per_sec[idx] = speed
            else:
                dvx, dvy = self.display_velocity[idx]
                self.display_velocity[idx] = (dvx * 0.9, dvy * 0.9)
                self.speed_ft_per_sec[idx] *= 0.9

    def _decay_velocity(self, idx: int):
        vx, vy = self.smoothed_velocity[idx]
        self.smoothed_velocity[idx] = (vx * 0.92, vy * 0.92)
        dvx, dvy = self.display_velocity[idx]
        self.display_velocity[idx] = (dvx * 0.92, dvy * 0.92)
        self.speed_ft_per_sec[idx] *= 0.92

    def _smooth_position(
        self, idx: int, new_pos: Tuple[float, float]
    ) -> Tuple[float, float]:
        if self.smoothed_pos[idx] is None:
            return new_pos
        old = self.smoothed_pos[idx]
        return (
            POSITION_ALPHA * new_pos[0] + (1 - POSITION_ALPHA) * old[0],
            POSITION_ALPHA * new_pos[1] + (1 - POSITION_ALPHA) * old[1],
        )

    def _split_merged_box(
        self, det: Tuple[float, float, float, float]
    ) -> List[Tuple[float, float, float, float]]:
        ax, ay, box_size, conf = det

        if self.smoothed_pos[0] is None or self.smoothed_pos[1] is None:
            return [det]

        p0, p1 = self.smoothed_pos[0], self.smoothed_pos[1]
        if self._distance(p0, p1) < MIN_SPLIT_DISTANCE_FT:
            return [det]

        mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        if self._distance((ax, ay), mid) > box_size:
            return [det]

        return [
            (p0[0], p0[1], box_size / 2, conf * 0.7),
            (p1[0], p1[1], box_size / 2, conf * 0.7),
        ]

    def filter_detections(
        self, boxes, confidences
    ) -> List[Tuple[float, float, float, float]]:
        valid = []

        for box, conf in zip(boxes, confidences):
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = box[:4]
            w, h = x2 - x1, y2 - y1
            area = w * h
            aspect = w / h if h > 0 else 0

            if conf < MIN_CONFIDENCE:
                continue
            if area < MIN_BOX_AREA_PX or area > MAX_BOX_AREA_PX:
                continue
            if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ax, ay = pixel_to_arena(cx, cy, self.H)

            ax1, ay1 = pixel_to_arena(x1, y1, self.H)
            ax2, ay2 = pixel_to_arena(x2, y2, self.H)
            box_size = np.sqrt((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2)

            if -5 <= ax <= ARENA_WIDTH_FT + 5 and -5 <= ay <= ARENA_HEIGHT_FT + 5:
                ax = np.clip(ax, 0, ARENA_WIDTH_FT)
                ay = np.clip(ay, 0, ARENA_HEIGHT_FT)
                valid.append((ax, ay, box_size, float(conf)))

        valid.sort(key=lambda x: x[3], reverse=True)
        return valid

    def update(self, boxes, confidences) -> List[Track]:
        self.frame_count += 1
        detections = self.filter_detections(boxes, confidences)

        self._last_robot_distance = self._get_robot_distance()
        robots_close = self._last_robot_distance < CLOSE_ROBOT_DIST_FT
        self._in_contact = self._last_robot_distance < CLOSE_PROXIMITY_FT

        if not self.initialized and self.auto_init and len(detections) >= 2:
            d0 = detections[0]
            for d1 in detections[1:]:
                if self._distance((d0[0], d0[1]), (d1[0], d1[1])) > 3.0:
                    for i, d in enumerate([d0, d1]):
                        self.tracks[i] = self._create_track(i, d[0], d[1])
                        self.last_detection_pos[i] = (d[0], d[1])
                        self.smoothed_pos[i] = (d[0], d[1])
                        self.typical_box_size[i] = d[2]
                    self.initialized = True
                    print(
                        f"Auto-init: USC at ({d0[0]:.1f}, {d0[1]:.1f}), ENEMY at ({d1[0]:.1f}, {d1[1]:.1f})"
                    )
                    break
            return self.get_active_tracks()

        if not self.initialized:
            return self.get_active_tracks()

        processed_dets = []
        for det in detections:
            if robots_close and self._is_merged_box(det[2]):
                processed_dets.extend(self._split_merged_box(det))
            else:
                processed_dets.append(det)

        if not processed_dets:
            for i in range(2):
                if self.tracks[i] is not None:
                    self.frames_since_detection[i] += 1
                    self._decay_velocity(i)
            return self.get_active_tracks()

        # Assignment
        threshold = (
            CONTACT_ASSIGNMENT_DIST_FT if robots_close else NORMAL_ASSIGNMENT_DIST_FT
        )
        assignment = [None, None]
        used_dets = set()

        for i in range(2):
            if self.smoothed_pos[i] is None:
                continue

            best_dist = float("inf")
            best_j = None

            for j, det in enumerate(processed_dets):
                if j in used_dets:
                    continue
                dist = self._distance(self.smoothed_pos[i], (det[0], det[1]))
                if dist < best_dist and dist < threshold:
                    if (
                        dist <= self.max_movement_per_frame
                        or self.frames_since_detection[i] > 5
                    ):
                        best_dist = dist
                        best_j = j

            if best_j is not None:
                assignment[i] = best_j
                used_dets.add(best_j)

        # Update track
        for i in range(2):
            if self.tracks[i] is None:
                continue

            if assignment[i] is not None:
                det = processed_dets[assignment[i]]
                raw_pos = (det[0], det[1])

                self._update_velocity(i, raw_pos)
                smoothed = self._smooth_position(i, raw_pos)

                measurement = np.array([[smoothed[0]], [smoothed[1]]], dtype=np.float32)
                self.tracks[i].kf.predict()
                self.tracks[i].kf.correct(measurement)
                self.tracks[i].age = 0

                self.last_detection_pos[i] = raw_pos
                self.smoothed_pos[i] = smoothed
                self.frames_since_detection[i] = 0

                if det[3] > 0.5:
                    if self.typical_box_size[i] is None:
                        self.typical_box_size[i] = det[2]
                    else:
                        self.typical_box_size[i] = (
                            0.9 * self.typical_box_size[i] + 0.1 * det[2]
                        )
            else:
                self.frames_since_detection[i] += 1
                self._decay_velocity(i)

        return self.get_active_tracks()

    def get_active_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t is not None]

    def get_display_velocity(self, idx: int) -> Tuple[float, float]:
        return self.display_velocity[idx]

    def is_in_contact_mode(self) -> bool:
        return self._in_contact

    def set_robot_position(self, robot_idx: int, px: int, py: int):
        ax, ay = pixel_to_arena(px, py, self.H)
        ax = np.clip(ax, 0, ARENA_WIDTH_FT)
        ay = np.clip(ay, 0, ARENA_HEIGHT_FT)

        if self.tracks[robot_idx] is None:
            self.tracks[robot_idx] = self._create_track(robot_idx, ax, ay)
        else:
            self.tracks[robot_idx].kf.statePost = np.array(
                [[ax], [ay], [0], [0]], dtype=np.float32
            )
            self.tracks[robot_idx].kf.statePre = np.array(
                [[ax], [ay], [0], [0]], dtype=np.float32
            )
            self.tracks[robot_idx].kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.last_detection_pos[robot_idx] = (ax, ay)
        self.smoothed_pos[robot_idx] = (ax, ay)
        self.raw_velocity[robot_idx] = (0, 0)
        self.smoothed_velocity[robot_idx] = (0, 0)
        self.display_velocity[robot_idx] = (0, 0)
        self.velocity_history[robot_idx].clear()
        self.speed_ft_per_sec[robot_idx] = 0.0
        self.frames_since_detection[robot_idx] = 0

        if self.tracks[0] is not None and self.tracks[1] is not None:
            self.initialized = True

        print(f"{self.labels[robot_idx]} set to ({ax:.1f}, {ay:.1f}) ft")

    def swap_labels(self):
        self.tracks[0], self.tracks[1] = self.tracks[1], self.tracks[0]

        for i, track in enumerate(self.tracks):
            if track is not None:
                track.id = i
                track.color = self.colors[i]

        for attr in [
            "last_detection_pos",
            "smoothed_pos",
            "raw_velocity",
            "smoothed_velocity",
            "display_velocity",
            "speed_ft_per_sec",
            "typical_box_size",
            "frames_since_detection",
        ]:
            lst = getattr(self, attr)
            lst[0], lst[1] = lst[1], lst[0]

        self.velocity_history[0], self.velocity_history[1] = (
            self.velocity_history[1],
            self.velocity_history[0],
        )

        print("Labels swapped: USC <-> ENEMY")

    def reset(self):
        self.tracks = [None, None]
        self.last_detection_pos = [None, None]
        self.smoothed_pos = [None, None]
        self.raw_velocity = [(0, 0), (0, 0)]
        self.smoothed_velocity = [(0, 0), (0, 0)]
        self.display_velocity = [(0, 0), (0, 0)]
        self.velocity_history = [
            deque(maxlen=VELOCITY_HISTORY_SIZE),
            deque(maxlen=VELOCITY_HISTORY_SIZE),
        ]
        self.speed_ft_per_sec = [0.0, 0.0]
        self.typical_box_size = [None, None]
        self.frames_since_detection = [0, 0]
        self.initialized = False
        self.frame_count = 0
        self._in_contact = False
        print("Tracker reset")

    def pixel_to_grid_cell(self, px: int, py: int) -> Tuple[int, int]:
        ax, ay = pixel_to_arena(px, py, self.H)
        ax = np.clip(ax, 0, ARENA_WIDTH_FT - 0.01)
        ay = np.clip(ay, 0, ARENA_HEIGHT_FT - 0.01)
        return int(ax / CELL_SIZE_FT), int(ay / CELL_SIZE_FT)

    def toggle_hazard(self, px: int, py: int):
        cx, cy = self.pixel_to_grid_cell(px, py)
        if 0 <= cx < GRID_CELLS_PER_SIDE and 0 <= cy < GRID_CELLS_PER_SIDE:
            self.hazard_grid[cy, cx] = not self.hazard_grid[cy, cx]
            print(
                f"Cell ({cx}, {cy}): {'HAZARD' if self.hazard_grid[cy, cx] else 'CLEAR'}"
            )

    def clear_hazards(self):
        self.hazard_grid.fill(False)
        print("Hazards cleared")
