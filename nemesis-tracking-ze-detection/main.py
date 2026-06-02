import argparse
import threading
import time as pytime
from collections import deque

import cv2
import numpy as np

from config import HIGH_PERFORMANCE_MODE
from display_thread import display_thread_func
from drawing_utils import calculate_distance
from tracker import RobotTracker

DEFAULT_STREAM_URL = "rtsp://192.168.68.51:8554/stream?rtsp_transport=tcp"
LIVESTREAM_MAX_WIDTH = 1280


def check_cuda_available():
    try:
        import torch

        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda"
    except ImportError:
        pass
    print("Using CPU")
    return "cpu"


def extract_tracker_state(tracker):
    positions = [None, None]
    velocities = [(0, 0), (0, 0)]
    speeds = [0.0, 0.0]

    for track in tracker.get_active_tracks():
        idx = track.id
        positions[idx] = track.position
        velocities[idx] = tracker.get_display_velocity(idx)
        speeds[idx] = tracker.speed_ft_per_sec[idx]

    distance = 0.0
    if positions[0] is not None and positions[1] is not None:
        distance = calculate_distance(positions[0], positions[1])

    return positions, velocities, speeds, distance


def process_video(
    video_path: str,
    model_path: str,
    show_live: bool = True,
    manual_init: bool = False,
    yolo_skip: int = 2,
    no_diff: bool = False,
):
    device = check_cuda_available()

    from ultralytics import YOLO

    print(f"Loading model {model_path}...")
    model = YOLO(model_path)
    if device == "cuda":
        model.to("cuda")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

    tracker = RobotTracker(auto_init=not manual_init)
    tracker.set_fps(fps)

    max_history = 300
    speed_history = [deque(maxlen=max_history), deque(maxlen=max_history)]
    distance_history = deque(maxlen=max_history)
    time_history = deque(maxlen=max_history)

    frame_buffer = []
    stop_event = threading.Event()
    inference_done = [False]

    if show_live:
        display_thread = threading.Thread(
            target=display_thread_func,
            args=(
                frame_buffer,
                frame_width,
                frame_height,
                stop_event,
                fps,
                no_diff,
                tracker,
                speed_history,
                distance_history,
                time_history,
                inference_done,
            ),
            daemon=True,
        )
        display_thread.start()

    frame_num = 0
    inference_count = 0
    start_time = pytime.perf_counter()
    boxes, confidences = [], []

    print("\n=== Controls ===")
    print("SPACE: Pause | R: Reset | F: Swap | H: Hazard | C: Clear | Q: Quit")
    print("SHIFT+LClick: Set USC | SHIFT+RClick: Set ENEMY")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            if frame_num % yolo_skip == 0:
                results = model(frame, verbose=False)[0]
                boxes = (
                    results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else []
                )
                confidences = (
                    results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else []
                )
                inference_count += 1
                tracker.update(boxes, confidences)

            elapsed = pytime.perf_counter() - start_time
            infer_fps = inference_count / elapsed if elapsed > 0 else 0

            positions, velocities, speeds, distance = extract_tracker_state(tracker)

            time_history.append(frame_num / fps)
            for i in range(2):
                speed_history[i].append(tracker.speed_ft_per_sec[i])
            distance_history.append(distance)

            frame_data = {
                "frame": frame.copy(),
                "frame_num": frame_num,
                "positions": positions,
                "velocities": velocities,
                "speeds": speeds,
                "labels": tracker.labels.copy(),
                "colors": tracker.colors.copy(),
                "H_inv": tracker.H_inv,
                "distance": distance,
                "infer_fps": infer_fps,
                "boxes": list(boxes) if len(boxes) > 0 else [],
                "confidences": list(confidences) if len(confidences) > 0 else [],
                "in_contact_mode": tracker.is_in_contact_mode(),
            }
            frame_buffer.append(frame_data)

            if frame_num % 500 == 0:
                print(
                    f"Processed {frame_num}/{total_frames} | Infer: {infer_fps:.0f} FPS"
                )

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        cap.release()
        inference_done[0] = True
        print(f"\nInference complete: {frame_num} frames")

        if show_live:
            while not stop_event.is_set():
                pytime.sleep(0.1)

    print("Done.")


def process_livestream(
    stream_url: str,
    model_path: str,
    manual_init: bool = False,
    yolo_skip: int = 1,
    no_diff: bool = True,
):
    device = check_cuda_available()

    from ultralytics import YOLO

    print(f"Loading model {model_path}...")
    model = YOLO(model_path)
    if device == "cuda":
        model.to("cuda")

    print(f"Connecting to stream: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Could not open stream: {stream_url}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    scale = 1.0
    if frame_width > LIVESTREAM_MAX_WIDTH:
        scale = LIVESTREAM_MAX_WIDTH / frame_width
        frame_width = int(frame_width * scale)
        frame_height = int(frame_height * scale)

    tracker = RobotTracker(auto_init=not manual_init)
    tracker.set_fps(fps)

    from display_views import (
        create_motion_view,
        create_sim_view,
        create_stats_view_minimal,
        create_video_view,
    )
    from timing import FrameTimer

    timer = FrameTimer()
    cv2.namedWindow("Robot Tracking - LIVE")

    hazard_mode = False
    paused = False
    display_scale = 1.0

    def mouse_callback(event, x, y, flags, param):
        nonlocal hazard_mode
        real_x = int(x / display_scale)
        real_y = int(y / display_scale)

        if real_x >= frame_width or real_y >= frame_height:
            return

        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            tracker.set_robot_position(0, real_x, real_y)
        elif event == cv2.EVENT_RBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            tracker.set_robot_position(1, real_x, real_y)
        elif event == cv2.EVENT_LBUTTONDOWN and hazard_mode:
            tracker.toggle_hazard(real_x, real_y)

    cv2.setMouseCallback("Robot Tracking - LIVE", mouse_callback)

    bg_subtractor = None
    motion_accumulator = None
    if not no_diff:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )

    frame_num = 0
    boxes, confidences = [], []
    last_frame_time = pytime.perf_counter()

    print("\n=== LIVESTREAM MODE ===")
    print("SPACE: Pause | R: Reset | F: Swap | H: Hazard | Q: Quit")

    try:
        while True:
            timer.start_frame()

            if paused:
                key = cv2.waitKey(50) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key == ord("q"):
                    break
                continue

            ret, frame = cap.read()
            if not ret:
                cap.release()
                pytime.sleep(1.0)
                cap = cv2.VideoCapture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            frame_num += 1

            if scale < 1.0:
                frame = cv2.resize(frame, (frame_width, frame_height))

            now = pytime.perf_counter()
            dt = now - last_frame_time
            last_frame_time = now
            tracker.set_dt(dt)

            timer.mark("grab")

            if frame_num % yolo_skip == 0:
                results = model(frame, verbose=False)[0]
                boxes = (
                    results.boxes.xyxy.cpu().numpy() if len(results.boxes) > 0 else []
                )
                confidences = (
                    results.boxes.conf.cpu().numpy() if len(results.boxes) > 0 else []
                )
                tracker.update(boxes, confidences)

            timer.mark("yolo")

            positions, velocities, speeds, distance = extract_tracker_state(tracker)

            timer.mark("track")

            video_view = create_video_view(
                frame,
                positions,
                velocities,
                tracker.labels,
                tracker.colors,
                tracker.H_inv,
                boxes,
                confidences,
            )

            sim_view = create_sim_view(
                positions,
                velocities,
                tracker.labels,
                tracker.colors,
                (frame_height, frame_width),
                tracker.H_inv,
                distance,
                hazard_grid=tracker.hazard_grid,
            )

            stats_view = create_stats_view_minimal(
                (frame_height, frame_width),
                frame_num,
                frame_num,
                frame_num * dt,
                timer.get_fps(),
                distance,
                positions,
                speeds,
                tracker.labels,
                tracker.colors,
                timer,
            )

            if no_diff or bg_subtractor is None:
                diff_view = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                cv2.putText(
                    diff_view,
                    "LIVE - NO MOTION",
                    (frame_width // 2 - 80, frame_height // 2),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (80, 80, 80),
                    1,
                )
            else:
                diff_view, motion_accumulator = create_motion_view(
                    frame,
                    bg_subtractor,
                    motion_accumulator,
                    (frame_height, frame_width),
                )

            timer.mark("viz")

            status = "LIVE"
            if tracker.is_in_contact_mode():
                status += " [CONTACT]"
            if hazard_mode:
                status += " [HAZARD]"
            cv2.putText(
                video_view,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            top = np.hstack([video_view, diff_view])
            bottom = np.hstack([sim_view, stats_view])
            combined = np.vstack([top, bottom])

            if combined.shape[1] > 1920:
                display_scale = 1920 / combined.shape[1]
                combined = cv2.resize(
                    combined, None, fx=display_scale, fy=display_scale
                )
            else:
                display_scale = 1.0

            cv2.imshow("Robot Tracking - LIVE", combined)
            timer.mark("display")
            timer.end_frame()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = True
            elif key == ord("r"):
                tracker.reset()
            elif key == ord("f"):
                tracker.swap_labels()
            elif key == ord("h"):
                hazard_mode = not hazard_mode
            elif key == ord("c"):
                tracker.clear_hazards()

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Robot Tracking")
    parser.add_argument("--video", "-v", help="Video path (omit for livestream)")
    parser.add_argument("--stream", "-s", default=DEFAULT_STREAM_URL)
    parser.add_argument("--model", "-m", required=True, help="YOLO model path")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-diff", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--yolo-skip", type=int, default=None)

    args = parser.parse_args()

    if args.video:
        yolo_skip = args.yolo_skip if args.yolo_skip else 2
        process_video(
            args.video,
            args.model,
            not args.no_display,
            args.manual,
            yolo_skip,
            args.no_diff,
        )
    else:
        yolo_skip = args.yolo_skip if args.yolo_skip else 1
        process_livestream(args.stream, args.model, args.manual, yolo_skip, True)


if __name__ == "__main__":
    main()
