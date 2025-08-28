"""
workers_sync.py  –  thread workers + TF ThreadPool
"""

from __future__ import annotations
import asyncio, threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Deque

import numpy as np
from scipy.signal import resample, find_peaks
from utils_fast import batch_prediction, get_volume, _optimise_single, get_volume_rt

# Config
s_freq = 5
TF_POOL = ThreadPoolExecutor(max_workers=2)
diagnosis = {0: "Afib", 1: "SVT", 2: "Sinus Bradycardi", 3: "Sinus Rhythm"}

# globals injected from app
frame_buffer: Deque[np.ndarray] | None = None
pred_event = threading.Event()
efhr_event = threading.Event()
wf_event = threading.Event()

import queue
send_queue = queue.PriorityQueue()  # (priority, timestamp, loop, ws, payload)


def _safe_send(loop, ws, payload: dict):
    loop.call_soon_threadsafe(asyncio.create_task, ws.send_json(payload))


def _fast_priority_send(loop, ws, payload, priority=0):
    """Ultra-fast priority-based sending with immediate dispatch"""
    
    # For ultra-smooth UI, send immediately without queueing for high-priority data
    if priority == 0:  # WfWorker - immediate send for smoothest UI
        loop.call_soon_threadsafe(asyncio.create_task, ws.send_json(payload))
    else:
        # Add to priority queue for lower priority data
        timestamp = time.time()
        send_queue.put((priority, timestamp, loop, ws, payload))


def _process_send_queue():
    """Process the send queue efficiently with minimal delays"""
    
    while True:
        try:
            priority, timestamp, loop, ws, payload = send_queue.get(timeout=0.01)
            
            loop.call_soon_threadsafe(asyncio.create_task, ws.send_json(payload))
            send_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            continue


threading.Thread(target=_process_send_queue, daemon=True).start()


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

            _fast_priority_send(
                self.loop,
                self.ws,
                {
                    "pid": "central",
                    "type": "prediction",
                    "diagnosis": diagnosis[int(logits.argmax())],
                    "confidence": float(logits.max()),
                },
                priority=3  # Lowest priority - predictions are less time-critical
            )


class EfHrWorker(threading.Thread):
    def __init__(self, ws, loop):
        super().__init__(daemon=True)
        self.ws = ws
        self.loop = loop
        self.last_send_time = 0

    def run(self):
        while True:
            efhr_event.wait(); efhr_event.clear()
            
            current_time = time.time()
            time_since_last = current_time - self.last_send_time
            if time_since_last < 0.8:
                sleep_time = min(0.1, 0.8 - time_since_last)  # Max 100ms sleep
                time.sleep(sleep_time)
                current_time = time.time()
            
            if frame_buffer is None or len(frame_buffer) < s_freq * 2:
                continue

            frames = np.asarray(frame_buffer)[-s_freq * 2 :]
            
            try:
                predicted = batch_prediction(frames)

                vol = get_volume(predicted)
                peaks, _ = find_peaks(vol)
                if peaks.size < 2:
                    continue

                print("[efhr] >>>condition met")

                rr = np.resize(np.diff(peaks), 10)[:10]
                ef = float(vol.max()); hr = float(s_freq * 60 / rr.mean())

                print({"pid": "central", "type": "efhr", "EF": ef, "HR": hr})

                _fast_priority_send(
                    self.loop,
                    self.ws,
                    {"pid": "central", "type": "efhr", "EF": ef, "HR": hr},
                    priority=1  # Higher priority than before (was 2)
                )
                
                # Update last send time
                self.last_send_time = current_time
                
            except Exception as e:
                print(f"[efhr] Error in processing: {e}")
                continue

class WfWorker(threading.Thread):
    def __init__(self, ws, loop):
        super().__init__(daemon=True)
        self.ws = ws
        self.loop = loop
        self.min = None
        self.last_send_time = 0

    def run(self):
        while True:
            wf_event.wait(); wf_event.clear()
            
            # Ultra-responsive: maximum 0.3 seconds interval (down from 0.5s)
            current_time = time.time()
            if current_time - self.last_send_time < 0.3:
                continue
                
            if frame_buffer is None:
                continue

            try:
                frame = np.asarray(frame_buffer)[-1]

                pos, height = _optimise_single(frame)

                predicted = np.concatenate(
                    [pos[:, None, :], height[:, None, None]], axis=-1
                ) 

                vol = float(get_volume_rt(predicted)[0])

                if self.min is not None:
                    self.min = np.min([self.min,vol])
                else:
                    self.min = vol
                    continue

                r_vol = vol/self.min

                print({"pid": "central", "type": "wf", "volume": r_vol})

                _fast_priority_send(
                    self.loop,
                    self.ws,
                    {"pid": "central", "type": "wf", "volume": r_vol},
                    priority=0  # Ultra-high priority - immediate send
                )
                
                # Update last send time
                self.last_send_time = current_time
                
            except Exception as e:
                print(f"[wf] Error in processing: {e}")
                # Continue without updating last_send_time to allow retry
                continue
