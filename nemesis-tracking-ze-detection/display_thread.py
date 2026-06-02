"""The yolo runs too fast on the video footage to display the visualizations in real time
instead, we store the inferences and their data from the inference in a buffer and play those back as fast as
we can sync it up to the real time pace
"""

import time as pytime

import cv2
import numpy as np

from config import HIGH_PERFORMANCE_MODE
from display_views import (
    create_motion_view,
    create_sim_view,
    create_stats_view_full,
    create_stats_view_minimal,
    create_video_view,
)
from drawing_utils import draw_grid_video, draw_hazard_cells_video
from timing import FrameTimer


def display_thread_func(
    frame_buffer,
    frame_width,
    frame_height,
    stop_event,
    video_fps,
    no_diff,
    tracker_ref,
    speed_history,
    distance_history,
    time_history,
    inference_done,
):
    cv2.namedWindow("Robot Tracking")

    hazard_mode = [False]
    display_scale = [1.0]
    paused = [False]

    def mouse_callback(event, x, y, flags, param):
        real_x = int(x / display_scale[0])
        real_y = int(y / display_scale[0])

        if real_x >= frame_width or real_y >= frame_height:
            return

        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            tracker_ref.set_robot_position(0, real_x, real_y)
        elif event == cv2.EVENT_RBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            tracker_ref.set_robot_position(1, real_x, real_y)
        elif event == cv2.EVENT_LBUTTONDOWN and hazard_mode[0]:
            tracker_ref.toggle_hazard(real_x, real_y)

    cv2.setMouseCallback("Robot Tracking", mouse_callback)

    bg_subtractor = None
    motion_accumulator = None
    if not no_diff and not HIGH_PERFORMANCE_MODE:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

    start_time = pytime.perf_counter()
    current_idx = 0
    timer = FrameTimer()

    while not stop_event.is_set():
        timer.start_frame()

        if paused[0]:
            key = cv2.waitKey(50) & 0xFF
            if key == ord(" "):
                paused[0] = False
                start_time = pytime.perf_counter() - (current_idx / video_fps)
            elif key == ord("q"):
                stop_event.set()
            elif key == ord("f"):
                tracker_ref.swap_labels()
            elif key == ord("r"):
                tracker_ref.reset()
            elif key == ord("h"):
                hazard_mode[0] = not hazard_mode[0]
            elif key == ord("c"):
                tracker_ref.clear_hazards()
            continue

        elapsed = pytime.perf_counter() - start_time
        target_idx = int(elapsed * video_fps)

        buffer_len = len(frame_buffer)
        if buffer_len == 0:
            pytime.sleep(0.01)
            continue

        if inference_done[0] and target_idx >= buffer_len:
            print(f"\nPlayback complete. {buffer_len} frames.")
            stop_event.set()
            break

        target_idx = min(target_idx, buffer_len - 1)

        if target_idx == current_idx and target_idx < buffer_len - 1:
            pytime.sleep(0.001)
            continue

        current_idx = target_idx

        try:
            data = frame_buffer[target_idx]
        except IndexError:
            pytime.sleep(0.01)
            continue

        frame = data["frame"]
        positions = data["positions"]
        velocities = data["velocities"]
        speeds = data.get("speeds", [0.0, 0.0])
        H_inv = data["H_inv"]
        distance = data["distance"]
        frame_num = data["frame_num"]
        infer_fps = data["infer_fps"]
        boxes = data.get("boxes", [])
        confidences = data.get("confidences", [])
        in_contact_mode = data.get("in_contact_mode", False)

        labels = tracker_ref.labels
        colors = tracker_ref.colors

        timer.mark("grab")

        video_view = create_video_view(
            frame, positions, velocities, labels, colors, H_inv, boxes, confidences
        )

        if hazard_mode[0]:
            draw_grid_video(video_view, H_inv)
            video_view = draw_hazard_cells_video(
                video_view, tracker_ref.hazard_grid, H_inv
            )

        sim_view = create_sim_view(
            positions,
            velocities,
            labels,
            colors,
            (frame_height, frame_width),
            H_inv,
            distance,
            hazard_grid=tracker_ref.hazard_grid,
        )

        if HIGH_PERFORMANCE_MODE:
            stats_view = create_stats_view_minimal(
                (frame_height, frame_width),
                frame_num,
                buffer_len,
                elapsed,
                infer_fps,
                distance,
                positions,
                speeds,
                labels,
                colors,
                timer,
            )
        else:
            stats_view = create_stats_view_full(
                (frame_height, frame_width),
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
                timer,
                speed_history,
                distance_history,
            )

        if HIGH_PERFORMANCE_MODE or no_diff:
            diff_view = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            msg = "PERF MODE" if HIGH_PERFORMANCE_MODE else "MOTION OFF"
            cv2.putText(
                diff_view,
                msg,
                (frame_width // 2 - 60, frame_height // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (80, 80, 80),
                1,
            )
        else:
            diff_view, motion_accumulator = create_motion_view(
                frame, bg_subtractor, motion_accumulator, (frame_height, frame_width)
            )

        timer.mark("viz")

        status = "TRACKING" if any(p is not None for p in positions) else "WAITING..."
        if in_contact_mode:
            status += " [CONTACT]"
        if hazard_mode[0]:
            status += " [HAZARD MODE]"
        if paused[0]:
            status += " [PAUSED]"
        cv2.putText(
            video_view,
            status,
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        top = np.hstack([video_view, diff_view])
        bottom = np.hstack([sim_view, stats_view])
        combined = np.vstack([top, bottom])

        if combined.shape[1] > 1920:
            scale = 1920 / combined.shape[1]
            display_scale[0] = scale
            combined = cv2.resize(combined, None, fx=scale, fy=scale)
        else:
            display_scale[0] = 1.0

        cv2.imshow("Robot Tracking", combined)
        timer.mark("display")
        timer.end_frame()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            stop_event.set()
        elif key == ord(" "):
            paused[0] = True
        elif key == ord("r"):
            tracker_ref.reset()
        elif key == ord("f"):
            tracker_ref.swap_labels()
        elif key == ord("h"):
            hazard_mode[0] = not hazard_mode[0]
        elif key == ord("c"):
            tracker_ref.clear_hazards()

    cv2.destroyAllWindows()
