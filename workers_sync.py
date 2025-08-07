"""
workers_sync.py  –  thread workers + TF ThreadPool
"""

from __future__ import annotations
import asyncio, threading
from concurrent.futures import ThreadPoolExecutor
from typing import Deque

import numpy as np
from scipy.signal import resample, find_peaks
from utils_fast import batch_prediction, get_volume

# Config
s_freq = 5
TF_POOL = ThreadPoolExecutor(max_workers=2)
diagnosis = {0: "Afib", 1: "SVT", 2: "Sinus Bradycardi", 3: "Sinus Rhythm"}

# globals injected from app
frame_buffer: Deque[np.ndarray] | None = None
pred_event = threading.Event()
efhr_event = threading.Event()


def _safe_send(loop, ws, payload: dict):
    loop.call_soon_threadsafe(asyncio.create_task, ws.send_json(payload))


class PredictionWorker(threading.Thread):
    def __init__(self, ws, loop, model):
        super().__init__(daemon=True)
        self.ws = ws
        self.loop = loop
        self.model = model

    def run(self):
        while True:
            pred_event.wait(); pred_event.clear()
            if frame_buffer is None or len(frame_buffer) < s_freq * 10:
                continue

            frames = np.asarray(frame_buffer, dtype=np.float32)
            predicted = batch_prediction(frames)

            vol = resample(get_volume(predicted), 500)
            peaks, _ = find_peaks(vol)
            if peaks.size < 2:
                continue

            print("[pred] >>>condition met")

            stem = np.zeros_like(vol, dtype=np.float32); stem[peaks] = 1.0
            cnn_in = stem[None, :, None]
            rr = np.resize(np.diff(peaks), 10)[:10]

            logits = TF_POOL.submit(
                lambda: self.model.predict([cnn_in, rr[np.newaxis, :]], verbose=0)
            ).result()[0]

            print({"pid": "central",
                    "type": "prediction",
                    "diagnosis": diagnosis[int(logits.argmax())],
                    "confidence": float(logits.max()),})

            _safe_send(
                self.loop,
                self.ws,
                {
                    "pid": "central",
                    "type": "prediction",
                    "diagnosis": diagnosis[int(logits.argmax())],
                    "confidence": float(logits.max()),
                },
            )


class EfHrWorker(threading.Thread):
    def __init__(self, ws, loop):
        super().__init__(daemon=True)
        self.ws = ws
        self.loop = loop

    def run(self):
        while True:
            efhr_event.wait(); efhr_event.clear()
            if frame_buffer is None or len(frame_buffer) < s_freq * 2:
                continue

            frames = np.asarray(frame_buffer)[-s_freq * 2 :]
            predicted = batch_prediction(frames)

            vol = get_volume(predicted)
            peaks, _ = find_peaks(vol)
            if peaks.size < 2:
                continue

            print("[efhr] >>>condition met")

            rr = np.resize(np.diff(peaks), 10)[:10]
            ef = float(vol.max()); hr = float(s_freq * 60 / rr.mean())

            print({"pid": "central", "type": "efhr", "EF": ef, "HR": hr})

            _safe_send(
                self.loop,
                self.ws,
                {"pid": "central", "type": "efhr", "EF": ef, "HR": hr},
            )
