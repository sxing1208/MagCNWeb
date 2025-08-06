from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
from utils import batch_prediction, get_volume
from scipy.signal import find_peaks
from tensorflow import keras

import asyncio
from collections import deque

from logging.handlers import RotatingFileHandler
import logging

from scipy.signal import resample

s_freq = 5

LOG_FILE = "app.log"
logging_lvl = logging.INFO

logger = logging.getLogger("APP")
logger.setLevel(logging_lvl)

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console_handler)


diagnosis = {0: 'Afib', 1: 'SVT', 2: 'Sinus Bradycardi', 3: 'Sinus Rhythm'}

model = keras.models.load_model("stitch_model.keras")

def run_prediction(model, cnn_input, rr_interval):
    return model.predict([cnn_input, rr_interval[np.newaxis, :]], verbose=0)

app = FastAPI()

frame_buffer = deque(maxlen=s_freq*10)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"pid":"central","type":"msg","message":"Connected to MagCN Cardiac Health Cloud!"})

    pred_task = asyncio.create_task(prediction_loop(websocket))
    efhr_task = asyncio.create_task(ef_hr_loop(websocket))

    while True:
        try:
            data = await websocket.receive_json()
            if data['pid'] == 'index':
                await handle_index(websocket, data)
            elif data['pid'] == 'device':
                await handle_device(websocket, data, frame_buffer)

        except Exception as e:
            logger.info("Client disconnected")
            pred_task.cancel()
            efhr_task.cancel()
            logger.warning(e)
            break

async def handle_index(websocket, data):
    logger.debug(data)

async def handle_device(websocket, data : dict, frame_list : deque):
    logger.debug(f"Received device data keys: {list(data.keys())}")
    logger.debug(f"Received device data values: {list(data.values())}")

    if data['key'] == "admin":
        values = np.array(data['value'].split(','), dtype=float).reshape((8,8))
        frame_list.append(values)

async def prediction_loop(websocket):
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(10)
        if len(frame_buffer) >= s_freq*10:
            logger.info("[pred] >>> Trigger condition met")
            try:
                frames = np.array(frame_buffer)
                logger.info("[pred] running prediction")
                predicted = await loop.run_in_executor(None, batch_prediction, frames)
                logger.debug("[pred] localization complete")

                volumes_base = get_volume(predicted)
                volumes = resample(volumes_base,500)
                peaks, _ = find_peaks(volumes)

                cnn_input = np.zeros(volumes.shape)
                cnn_input[peaks] = 1
                cnn_input = np.stack([cnn_input], axis=-1)

                rr_interval = np.diff(peaks)
                rr_interval = np.tile(rr_interval, int(np.ceil(10 / len(rr_interval)))+1)[:10]

                cnn_input = np.expand_dims(cnn_input, axis=0) 

                logits = await loop.run_in_executor(None, run_prediction, model, cnn_input, rr_interval)
                final_logits = logits[0]

                diag = diagnosis[np.argmax(final_logits)]

                logger.info({"pid":"central","type": "prediction", "diagnosis":diag, "confidence":float(np.max(final_logits))})

                await websocket.send_json({"pid":"central","type": "prediction", "diagnosis":diag, "confidence":float(np.max(final_logits))})

            except Exception as e:
                logger.warning(f"Prediction loop error: {e}")


async def ef_hr_loop(websocket):
    loop = asyncio.get_running_loop()
    while True:
        await asyncio.sleep(2)
        logger.debug(f"[efhr] Buffer length: {len(frame_buffer)}")
        if len(frame_buffer) >= s_freq*2:
            logger.info("[efhr] >>> Trigger condition met")
            try:
                frames = np.array(frame_buffer)[-s_freq*2:]
                logger.info("[efhr] running prediction")
                predicted = await loop.run_in_executor(None, batch_prediction, frames)
                logger.debug("[efhr] localization complete")

                volumes = get_volume(predicted)
                peaks, _ = find_peaks(volumes)

                rr_interval = np.diff(peaks)
                rr_interval = np.tile(rr_interval, int(np.ceil(10 / len(rr_interval)))+1)[:10]

                ef = np.max(volumes)
                hr = s_freq*60/np.mean(rr_interval)

                logger.info({"pid":"central","type": "efhr", "EF": ef, "HR": hr})

                await websocket.send_json({"pid":"central","type": "efhr", "EF": ef, "HR": hr})

            except Exception as e:
                logger.warning(f"efhr loop error: {e}")
