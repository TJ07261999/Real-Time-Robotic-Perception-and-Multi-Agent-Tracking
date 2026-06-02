import time
from collections import deque
from typing import Dict


class FrameTimer:

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.times: Dict[str, deque] = {}
        self.start_time = 0
        self.stage_start = 0

    def start_frame(self):
        self.start_time = time.perf_counter()
        self.stage_start = self.start_time

    def mark(self, stage: str):
        now = time.perf_counter()
        elapsed_ms = (now - self.stage_start) * 1000
        if stage not in self.times:
            self.times[stage] = deque(maxlen=self.window_size)
        self.times[stage].append(elapsed_ms)
        self.stage_start = now

    def end_frame(self):
        now = time.perf_counter()
        total_ms = (now - self.start_time) * 1000
        if "total" not in self.times:
            self.times["total"] = deque(maxlen=self.window_size)
        self.times["total"].append(total_ms)

    def get_avg(self, stage: str) -> float:
        if stage not in self.times or not self.times[stage]:
            return 0.0
        return sum(self.times[stage]) / len(self.times[stage])

    def get_fps(self) -> float:
        avg_total = self.get_avg("total")
        if avg_total <= 0:
            return 0.0
        return 1000.0 / avg_total
