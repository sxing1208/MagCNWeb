import asyncio
import websockets
import json
import numpy as np

s_freq = 5

FASTAPI_WS_URL = "ws://localhost:8000/ws"
NPY_FILE = "classification_model/st_data.npy"

async def push_frames():
    # Load frames
    data = np.load(NPY_FILE)  # shape: (num_frames, 8, 8) or (num_frames, 64)
    print(f"[Loader] Loaded array with shape {data.shape}")

    # Ensure each frame is flat 64 values
    if data.ndim == 3 and data.shape[1:] == (8, 8):
        frames = data.reshape(data.shape[0], -1)
    elif data.ndim == 2 and data.shape[1] == 64:
        frames = data
    else:
        raise ValueError(f"Unexpected data shape {data.shape}, expected (N,8,8) or (N,64)")

    while True:  # reconnect loop
        try:
            async with websockets.connect(
                FASTAPI_WS_URL,
                ping_interval=20,   # send ping every 20s
                ping_timeout=10     # fail if no pong in 10s
            ) as websocket:

                print(f"[WS] Connected to {FASTAPI_WS_URL}")

                # Start a background task to read incoming messages (keeps pings alive)
                async def keepalive_reader():
                    try:
                        async for _ in websocket:
                            pass  # ignore incoming data
                    except websockets.ConnectionClosed:
                        pass

                reader_task = asyncio.create_task(keepalive_reader())

                # Send frames forever
                while True:
                    for i, frame in enumerate(frames):
                        payload = {
                            "pid": "device",
                            "key": "admin",
                            "value": ",".join(map(str, frame))
                        }
                        await websocket.send(json.dumps(payload))
                        print(f"[WS] Sent frame {i+1}/{len(frames)}")
                        await asyncio.sleep(1 / s_freq)

                await reader_task  # keep running reader

        except websockets.ConnectionClosed:
            print("[WS] Connection closed, retrying in 5s...")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(push_frames())
