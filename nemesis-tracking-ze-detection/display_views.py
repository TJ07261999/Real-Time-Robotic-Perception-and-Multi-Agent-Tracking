import cv2
import numpy as np

from config import ARENA_HEIGHT_FT, ARENA_WIDTH_FT, CALIBRATION_CLICKS, COLORS
from drawing_utils import (
    create_arena_to_sim_transform,
    draw_distance_line,
    draw_grid_sim,
    draw_hazard_cells_sim,
    draw_rounded_rect,
    draw_text_styled,
    draw_velocity_arrow,
    get_distance_color,
)
from homography import arena_to_pixel


def create_sim_view(
    positions,
    velocities,
    labels,
    colors,
    frame_size,
    H_inv,
    distance=None,
    hazard_grid=None,
):
    height, width = frame_size
    sim = np.full((height, width, 3), 240, dtype=np.uint8)

    arena_to_sim = create_arena_to_sim_transform(width, height)

    # Hazards
    if hazard_grid is not None:
        draw_hazard_cells_sim(sim, hazard_grid, arena_to_sim)

    # Arena boundary
    corners = [
        arena_to_sim(0, 0),
        arena_to_sim(ARENA_WIDTH_FT, 0),
        arena_to_sim(ARENA_WIDTH_FT, ARENA_HEIGHT_FT),
        arena_to_sim(0, ARENA_HEIGHT_FT),
    ]
    cv2.polylines(sim, [np.array(corners)], True, (0, 0, 0), 2)

    # Grid
    draw_grid_sim(sim, arena_to_sim)

    # Distance line
    if (
        positions[0] is not None
        and positions[1] is not None
        and distance is not None
        and distance < 15
    ):
        p0 = arena_to_sim(positions[0][0], positions[0][1])
        p1 = arena_to_sim(positions[1][0], positions[1][1])
        draw_distance_line(sim, p0, p1, distance)

    # Robots
    for pos, vel, label, color in zip(positions, velocities, labels, colors):
        if pos is None:
            continue
        sx, sy = arena_to_sim(pos[0], pos[1])

        cv2.circle(sim, (sx, sy), 12, color, -1)
        cv2.circle(sim, (sx, sy), 12, (0, 0, 0), 2)

        draw_velocity_arrow(sim, (sx, sy), vel, color)

        cv2.putText(
            sim, label, (sx + 15, sy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
        cv2.putText(
            sim,
            f"({pos[0]:.1f}, {pos[1]:.1f})",
            (sx + 15, sy + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (80, 80, 80),
            1,
        )

    # Title
    cv2.rectangle(sim, (5, 5), (95, 30), (0, 0, 0), -1)
    cv2.putText(
        sim, "GRID SIM", (10, 24), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1
    )

    return sim


def create_video_view(
    frame, positions, velocities, labels, colors, H_inv, boxes, confidences
):
    video_view = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    # Detection boxes
    for box, conf in zip(boxes, confidences):
        if len(box) >= 4:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(video_view, (x1, y1), (x2, y2), (128, 128, 128), 1)
            cv2.putText(
                video_view,
                f"{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
            )

    # Robots
    for pos, vel, label, color in zip(positions, velocities, labels, colors):
        if pos is None:
            continue
        px, py = arena_to_pixel(pos[0], pos[1], H_inv)
        px = max(0, min(frame_w, px))
        py = max(0, min(frame_h, py))

        cv2.circle(video_view, (px, py), 20, color, 2)
        cv2.circle(video_view, (px, py), 16, color, -1)

        label_x = px + 25
        label_y = py
        cv2.putText(
            video_view,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        # Velocity arrow
        vx, vy = vel
        speed = np.sqrt(vx * vx + vy * vy)
        if speed > 0.5:
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            arrow_start_x = label_x + lw + 10
            arrow_start_y = label_y - lh // 2
            arrow_len = max(30, min(80, speed * 3.0))
            arrow_end_x = int(arrow_start_x + (vx / speed) * arrow_len)
            arrow_end_y = int(arrow_start_y - (vy / speed) * arrow_len)
            cv2.arrowedLine(
                video_view,
                (arrow_start_x, arrow_start_y),
                (arrow_end_x, arrow_end_y),
                color,
                2,
                tipLength=0.3,
            )

    # Calibration square
    for i, pt in enumerate(CALIBRATION_CLICKS):
        next_pt = CALIBRATION_CLICKS[(i + 1) % 4]
        cv2.line(video_view, pt, next_pt, (180, 140, 80), 1)

    return video_view


def create_stats_view_full(
    frame_size,
    frame_num,
    buffer_len,
    elapsed,
    infer_fps,
    distance,
    positions,
    speeds,
    labels,
    colors,
    video_fps,
    timer=None,
    speed_history=None,
    distance_history=None,
):
    height, width = frame_size
    stats = np.full((height, width, 3), 20, dtype=np.uint8)

    ACCENT_BLUE = COLORS["ACCENT_BLUE"]
    ACCENT_CYAN = COLORS["ACCENT_CYAN"]
    TEXT_WHITE = COLORS["TEXT_WHITE"]
    TEXT_DIM = COLORS["TEXT_DIM"]

    # Title
    cv2.rectangle(stats, (0, 0), (width, 55), (35, 35, 35), -1)
    cv2.line(stats, (0, 55), (width, 55), ACCENT_BLUE, 1, cv2.LINE_AA)
    draw_text_styled(stats, "COMBAT TRACKER", (20, 38), 0.9, TEXT_WHITE, 2)

    # Controls
    ctrl_y = 70
    draw_rounded_rect(
        stats, (15, ctrl_y), (width - 15, ctrl_y + 220), (30, 30, 30), 1, 10
    )
    draw_text_styled(stats, "CONTROLS", (25, ctrl_y + 25), 0.55, ACCENT_BLUE, 1)

    controls = [
        ("SPACE", "Pause"),
        ("R", "Reset"),
        ("F", "Swap"),
        ("H", "Hazard"),
        ("C", "Clear"),
        ("Sh+L", "Set USC"),
        ("Sh+R", "Set ENM"),
        ("Q", "Quit"),
    ]

    for i, (key, desc) in enumerate(controls):
        col = 25 if i < 4 else width // 2 + 10
        y_pos = ctrl_y + 60 + (i % 4) * 40
        key_w = 90
        cv2.rectangle(
            stats, (col, y_pos - 22), (col + key_w, y_pos + 8), (50, 50, 50), -1
        )
        draw_rounded_rect(
            stats, (col, y_pos - 22), (col + key_w, y_pos + 8), ACCENT_CYAN, 1, 5
        )
        draw_text_styled(
            stats, key, (col + 8, y_pos), 0.55, TEXT_WHITE, 1, shadow=False
        )
        draw_text_styled(stats, desc, (col + key_w + 12, y_pos), 0.6, TEXT_DIM, 1)

    # Telemetry
    stats_y = ctrl_y + 235
    draw_rounded_rect(
        stats, (15, stats_y), (width - 15, stats_y + 130), (30, 30, 30), 1, 10
    )
    draw_text_styled(stats, "TELEMETRY", (25, stats_y + 25), 0.5, ACCENT_BLUE, 1)

    # Robot 0
    if positions[0] is not None:
        cv2.circle(stats, (35, stats_y + 55), 6, colors[0], -1, cv2.LINE_AA)
        draw_text_styled(stats, labels[0], (50, stats_y + 60), 0.6, colors[0], 1)
        draw_text_styled(
            stats,
            f"POS: {positions[0][0]:.1f}, {positions[0][1]:.1f}",
            (25, stats_y + 85),
            0.4,
            TEXT_DIM,
            1,
        )
        draw_text_styled(
            stats, f"{speeds[0]:.1f}", (25, stats_y + 115), 1.0, TEXT_WHITE, 2
        )
        draw_text_styled(stats, "ft/s", (85, stats_y + 115), 0.4, TEXT_DIM, 1)

    # Robot 1
    ex = width // 2 + 10
    if positions[1] is not None:
        cv2.circle(stats, (ex + 10, stats_y + 55), 6, colors[1], -1, cv2.LINE_AA)
        draw_text_styled(stats, labels[1], (ex + 25, stats_y + 60), 0.6, colors[1], 1)
        draw_text_styled(
            stats,
            f"POS: {positions[1][0]:.1f}, {positions[1][1]:.1f}",
            (ex, stats_y + 85),
            0.4,
            TEXT_DIM,
            1,
        )
        draw_text_styled(
            stats, f"{speeds[1]:.1f}", (ex, stats_y + 115), 1.0, TEXT_WHITE, 2
        )
        draw_text_styled(stats, "ft/s", (ex + 60, stats_y + 115), 0.4, TEXT_DIM, 1)

    # Distance
    dist_y = stats_y + 140
    draw_rounded_rect(
        stats, (15, dist_y), (width - 15, dist_y + 35), (40, 40, 40), -1, 5
    )
    draw_text_styled(stats, "RANGE", (25, dist_y + 24), 0.4, TEXT_DIM, 1)
    draw_text_styled(
        stats, f"{distance:.1f} ft", (width - 120, dist_y + 24), 0.7, ACCENT_CYAN, 2
    )

    # Time
    time_y = dist_y + 45
    time_sec = frame_num / video_fps if video_fps > 0 else 0
    mins, secs = int(time_sec // 60), time_sec % 60
    draw_text_styled(
        stats, f"TIME: {mins:02d}:{secs:05.2f}", (25, time_y + 22), 0.5, TEXT_WHITE, 1
    )
    draw_text_styled(
        stats,
        f"Frame {frame_num}/{buffer_len}",
        (width - 150, time_y + 22),
        0.4,
        TEXT_DIM,
        1,
    )

    # Timing
    timing_y = time_y + 40
    draw_rounded_rect(
        stats, (15, timing_y), (width - 15, timing_y + 45), (25, 25, 25), 1, 5
    )

    if timer:
        disp_fps = timer.get_fps()
        realtime_pct = (disp_fps / video_fps * 100) if video_fps > 0 else 0
        fps_color = (0, 255, 100) if realtime_pct >= 100 else (0, 200, 255)
        draw_text_styled(
            stats, f"{disp_fps:.0f} FPS", (25, timing_y + 18), 0.5, fps_color, 1
        )
        draw_text_styled(
            stats, f"({realtime_pct:.0f}% RT)", (100, timing_y + 18), 0.4, TEXT_DIM, 1
        )
        draw_text_styled(
            stats,
            f"G:{timer.get_avg('grab'):.0f} V:{timer.get_avg('viz'):.0f} D:{timer.get_avg('display'):.0f}ms",
            (25, timing_y + 38),
            0.35,
            TEXT_DIM,
            1,
        )
    else:
        draw_text_styled(
            stats,
            f"Infer: {infer_fps:.0f} FPS",
            (25, timing_y + 25),
            0.5,
            (0, 255, 100),
            1,
        )

    # Graph
    graph_y = timing_y + 55
    graph_h = height - graph_y - 20

    if graph_h > 100 and distance_history is not None and len(distance_history) > 1:
        draw_rounded_rect(
            stats, (15, graph_y), (width - 15, graph_y + graph_h), (30, 30, 30), 1, 10
        )

        graph_x, graph_w = 25, width - 50
        inner_h = graph_h - 45
        inner_y = graph_y + 35

        draw_text_styled(
            stats, "ENGAGEMENT PROFILE", (graph_x, graph_y + 22), 0.45, ACCENT_BLUE, 1
        )

        for i in range(5):
            gy = inner_y + int(i * inner_h / 4)
            cv2.line(
                stats,
                (graph_x, gy),
                (graph_x + graph_w, gy),
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )

        points = []
        max_dist = 15.0
        hist_list = list(distance_history)

        for i, d in enumerate(hist_list):
            px = graph_x + int(i * graph_w / max(len(hist_list) - 1, 1))
            py = inner_y + inner_h - int((min(d, max_dist) / max_dist) * inner_h)
            py = max(inner_y, min(inner_y + inner_h, py))
            points.append((px, py))

        for i in range(1, len(points)):
            color = get_distance_color(hist_list[i])
            cv2.line(stats, points[i - 1], points[i], color, 2, cv2.LINE_AA)

    return stats


def create_stats_view_minimal(
    frame_size,
    frame_num,
    buffer_len,
    elapsed,
    infer_fps,
    distance,
    positions,
    speeds,
    labels,
    colors,
    timer=None,
):
    height, width = frame_size
    stats = np.full((height, width, 3), 30, dtype=np.uint8)

    y = 40
    cv2.putText(
        stats,
        f"Frame: {frame_num}/{buffer_len}",
        (20, y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (255, 255, 255),
        1,
    )
    y += 40
    cv2.putText(
        stats,
        f"Time: {elapsed:.1f}s",
        (20, y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (0, 255, 100),
        1,
    )
    y += 40
    cv2.putText(
        stats,
        f"Infer: {infer_fps:.0f} FPS",
        (20, y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (100, 255, 255),
        1,
    )
    y += 40
    cv2.putText(
        stats,
        f"Distance: {distance:.1f} ft",
        (20, y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (255, 200, 100),
        1,
    )

    y += 50
    for pos, speed, label, color in zip(positions, speeds, labels, colors):
        if pos is None:
            continue
        cv2.putText(
            stats,
            f"{label}: {speed:.1f} ft/s",
            (20, y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            color,
            1,
        )
        y += 35

    y += 20
    if timer:
        cv2.putText(
            stats,
            f"Display: {timer.get_fps():.0f} FPS",
            (20, y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (100, 200, 100),
            1,
        )
        y += 30
        cv2.putText(
            stats,
            f"G:{timer.get_avg('grab'):.0f} V:{timer.get_avg('viz'):.0f} D:{timer.get_avg('display'):.0f}ms",
            (20, y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

    return stats


def create_motion_view(frame, bg_subtractor, motion_accumulator, frame_size):
    height, width = frame_size

    if bg_subtractor is None:
        diff_view = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            diff_view,
            "MOTION DISABLED",
            (width // 2 - 100, height // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (80, 80, 80),
            1,
        )
        return diff_view, motion_accumulator

    fg_mask = bg_subtractor.apply(frame, learningRate=0.005)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    if motion_accumulator is None:
        motion_accumulator = fg_mask.astype(np.float32)
    else:
        motion_accumulator = motion_accumulator * 0.95
        motion_accumulator = np.maximum(motion_accumulator, fg_mask.astype(np.float32))

    display_mask = np.clip(motion_accumulator, 0, 255).astype(np.uint8)
    diff_view = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        diff_view,
        "MOTION TRACKING",
        (width - 200, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    return diff_view, motion_accumulator
