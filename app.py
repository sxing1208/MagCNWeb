"""
app.py  –  FastAPI WebSocket entry
"""

import asyncio, logging
from collections import deque
from typing import Deque

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow import keras

import workers_sync as ws

# logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("APP")

# shared structures
frame_buffer: Deque[np.ndarray] = deque(maxlen=ws.s_freq * 10)
ws.frame_buffer = frame_buffer  # inject into workers

model = keras.models.load_model("stitch_model.keras")

# FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws_socket: WebSocket):
    await ws_socket.accept()
    await ws_socket.send_json(
        {"pid": "central", "type": "msg", "message": "Connected!"}
    )

    loop = asyncio.get_running_loop()
    ws.PredictionWorker(ws_socket, loop, model).start()
    ws.EfHrWorker(ws_socket, loop).start()

    try:
        while True:
            data = await ws_socket.receive_json()
            if data.get("pid") == "device" and data.get("key") == "admin":
                frame = np.fromstring(data["value"], sep=",", dtype=np.float32).reshape(
                    8, 8
                )
                frame_buffer.append(frame)
                if len(frame_buffer) >= ws.s_freq * 10:
                    ws.pred_event.set()
                if len(frame_buffer) >= ws.s_freq * 2:
                    ws.efhr_event.set()
    except Exception as exc:
        logger.info("WebSocket closed: %s", exc)
