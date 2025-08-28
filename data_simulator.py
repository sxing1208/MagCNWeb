#!/usr/bin/env python3
"""
Data Simulator - Sends realistic ECG-like data to the server
This simulates a medical device sending frame data
"""
import asyncio
import json
import time
import numpy as np
import websockets

async def simulate_data_stream():
    """Simulate continuous ECG data stream"""
    print("🚀 Starting data simulation...")
    print("This will send realistic ECG frames to both endpoints")
    print("You should see prioritized data in the UI!")
    print("=" * 60)
    
    try:
        # Connect to the main endpoint that has both workers
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            print("✅ Connected to server")
            
            # Listen for server responses
            async def listen_for_responses():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        if data.get("type") == "efhr":
                            print(f"📊 EfHr Data: EF={data.get('EF', 'N/A'):.2f}, HR={data.get('HR', 'N/A'):.1f}")
                        elif data.get("type") == "msg":
                            print(f"💬 Server: {data.get('message', '')}")
                except websockets.exceptions.ConnectionClosed:
                    print("📡 Connection closed")
                except Exception as e:
                    print(f"⚠️  Listener error: {e}")
            
            # Start listener
            listener_task = asyncio.create_task(listen_for_responses())
            
            print("📤 Starting continuous data stream...")
            print("Press Ctrl+C to stop")
            print("-" * 40)
            
            frame_count = 0
            start_time = time.time()
            
            try:
                while True:
                    # Generate realistic ECG-like 8x8 frame
                    frame = np.random.uniform(0.2, 0.4, (8, 8)).astype(np.float32)
                    
                    # Add cardiac cycle pattern (simulates heartbeat)
                    cycle_position = (frame_count % 20) / 20.0  # 20-frame cycle
                    
                    if cycle_position < 0.1:  # QRS complex (main peak)
                        frame[2:6, 2:6] += 0.8
                        frame[3:5, 3:5] += 0.4
                    elif 0.15 < cycle_position < 0.25:  # T wave
                        frame[1:3, 4:6] += 0.3
                        frame[5:7, 2:4] += 0.25
                    elif 0.6 < cycle_position < 0.7:  # P wave  
                        frame[4:6, 1:3] += 0.2
                        frame[2:4, 5:7] += 0.15
                    
                    # Add some noise for realism
                    frame += np.random.uniform(-0.05, 0.05, (8, 8))
                    frame = np.clip(frame, 0, 1)  # Keep values in valid range
                    
                    # Send frame to server
                    payload = {
                        "pid": "device",
                        "key": "admin",
                        "value": ",".join(map(str, frame.flatten()))
                    }
                    
                    await websocket.send(json.dumps(payload))
                    frame_count += 1
                    
                    # Status update every 50 frames
                    if frame_count % 50 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"📈 Sent {frame_count} frames ({fps:.1f} fps)")
                    
                    # Send at ~10 FPS (realistic for medical devices)
                    await asyncio.sleep(0.1)
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping simulation...")
                print(f"📊 Total frames sent: {frame_count}")
                print(f"⏱️  Duration: {time.time() - start_time:.1f} seconds")
                
            finally:
                listener_task.cancel()
                try:
                    await asyncio.wait_for(listener_task, timeout=1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                    
    except ConnectionRefusedError:
        print("❌ Cannot connect to server!")
        print("Make sure the server is running: uvicorn app:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")

async def simulate_wf_endpoint():
    """Also connect to WfWorker endpoint to see its prioritized output"""
    print("\n🔄 Also connecting to WfWorker endpoint...")
    
    try:
        async with websockets.connect("ws://localhost:8000/ws2") as websocket:
            print("✅ Connected to WfWorker endpoint")
            
            # Listen for WfWorker responses
            async def listen_wf():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        if data.get("type") == "wf":
                            print(f"🎯 WF Data (Priority): volume={data.get('volume', 'N/A'):.4f}")
                        elif data.get("type") == "msg":
                            print(f"💬 WF Server: {data.get('message', '')}")
                except websockets.exceptions.ConnectionClosed:
                    pass
                except Exception as e:
                    print(f"⚠️  WF Listener error: {e}")
            
            listener_task = asyncio.create_task(listen_wf())
            
            # Send same data to WfWorker endpoint
            frame_count = 0
            try:
                while True:
                    # Generate frame (same pattern as main endpoint)
                    frame = np.random.uniform(0.3, 0.7, (8, 8)).astype(np.float32)
                    cycle_position = (frame_count % 15) / 15.0
                    
                    if cycle_position < 0.2:
                        frame[2:6, 2:6] += 0.4
                        frame[3:5, 3:5] += 0.2
                    
                    payload = {
                        "pid": "device", 
                        "key": "admin",
                        "value": ",".join(map(str, frame.flatten()))
                    }
                    
                    await websocket.send(json.dumps(payload))
                    frame_count += 1
                    await asyncio.sleep(0.1)
                    
            except KeyboardInterrupt:
                pass
            finally:
                listener_task.cancel()
                
    except Exception as e:
        print(f"⚠️  WF endpoint error: {e}")

async def main():
    """Run both data streams concurrently"""
    print("🌟 ECG Data Simulator for Priority System Testing")
    print("This will demonstrate WfWorker priority over EfHrWorker")
    print("=" * 60)
    
    # Run both simulators concurrently
    await asyncio.gather(
        simulate_data_stream(),
        simulate_wf_endpoint(),
        return_exceptions=True
    )

if __name__ == "__main__":
    asyncio.run(main())
