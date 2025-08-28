#!/usr/bin/env python3
"""
timing_test.py - Test the optimized threading system timing
"""

import asyncio
import json
import time
import numpy as np
import websockets
from collections import defaultdict

async def test_timing():
    """Test the timing constraints for WfWorker and EfHrWorker"""
    print("🚀 Testing Optimized Threading System")
    print("=" * 50)
    print("Target: WfWorker ≤0.5s, EfHrWorker ~1s")
    print("=" * 50)
    
    timing_data = defaultdict(list)
    message_count = defaultdict(int)
    
    try:
        # Connect to the server
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            print("✅ Connected to server")
            
            # Listen for responses and track timing
            async def listen_and_track():
                last_wf_time = None
                last_efhr_time = None
                
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        current_time = time.time()
                        
                        if data.get("type") == "wf":
                            message_count["wf"] += 1
                            if last_wf_time is not None:
                                interval = current_time - last_wf_time
                                timing_data["wf"].append(interval)
                                print(f"🎯 WF interval: {interval:.3f}s (target: ≤0.5s)")
                            last_wf_time = current_time
                            
                        elif data.get("type") == "efhr":
                            message_count["efhr"] += 1
                            if last_efhr_time is not None:
                                interval = current_time - last_efhr_time
                                timing_data["efhr"].append(interval)
                                print(f"📊 EfHr interval: {interval:.3f}s (target: ~1.0s)")
                            last_efhr_time = current_time
                            
                        elif data.get("type") == "msg":
                            print(f"💬 Server: {data.get('message', '')}")
                            
                except websockets.exceptions.ConnectionClosed:
                    print("📡 Connection closed")
                except Exception as e:
                    print(f"⚠️  Listener error: {e}")
            
            # Start listener
            listener_task = asyncio.create_task(listen_and_track())
            
            print("📤 Sending test data...")
            
            # Send realistic test data for 20 seconds
            frame_count = 0
            start_time = time.time()
            
            try:
                while time.time() - start_time < 20:  # Run for 20 seconds
                    # Generate realistic ECG-like 8x8 frame
                    frame = np.random.uniform(0.2, 0.4, (8, 8)).astype(np.float32)
                    
                    # Add cardiac cycle pattern
                    cycle_position = (frame_count % 20) / 20.0
                    
                    if cycle_position < 0.1:  # QRS complex
                        frame[2:6, 2:6] += 0.8
                        frame[3:5, 3:5] += 0.4
                    elif 0.15 < cycle_position < 0.25:  # T wave
                        frame[1:3, 4:6] += 0.3
                        frame[5:7, 2:4] += 0.25
                    
                    # Add noise for realism
                    frame += np.random.uniform(-0.05, 0.05, (8, 8))
                    frame = np.clip(frame, 0, 1)
                    
                    # Send frame
                    payload = {
                        "pid": "device",
                        "key": "admin",
                        "value": ",".join(map(str, frame.flatten()))
                    }
                    
                    await websocket.send(json.dumps(payload))
                    frame_count += 1
                    
                    # Debug: status update every 25 frames
                    if frame_count % 25 == 0:
                        print(f"📈 Sent {frame_count} frames, WF msgs: {message_count['wf']}, EfHr msgs: {message_count['efhr']}")
                    
                    # Send at ~10 FPS
                    await asyncio.sleep(0.1)
                    
            except KeyboardInterrupt:
                print(f"\n🛑 Test interrupted")
                
            finally:
                listener_task.cancel()
                try:
                    await asyncio.wait_for(listener_task, timeout=1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                    
            # Print results
            print("\n" + "=" * 50)
            print("📈 TIMING ANALYSIS RESULTS")
            print("=" * 50)
            
            if timing_data["wf"]:
                wf_avg = np.mean(timing_data["wf"])
                wf_max = np.max(timing_data["wf"])
                wf_violations = sum(1 for t in timing_data["wf"] if t > 0.5)
                print(f"🎯 WfWorker:")
                print(f"   Average interval: {wf_avg:.3f}s")
                print(f"   Maximum interval: {wf_max:.3f}s")
                print(f"   Target violations (>0.5s): {wf_violations}/{len(timing_data['wf'])}")
                print(f"   Messages received: {message_count['wf']}")
                
            if timing_data["efhr"]:
                efhr_avg = np.mean(timing_data["efhr"])
                efhr_min = np.min(timing_data["efhr"])
                efhr_max = np.max(timing_data["efhr"])
                print(f"📊 EfHrWorker:")
                print(f"   Average interval: {efhr_avg:.3f}s")
                print(f"   Min interval: {efhr_min:.3f}s")
                print(f"   Max interval: {efhr_max:.3f}s")
                print(f"   Messages received: {message_count['efhr']}")
                
            # Success criteria
            wf_success = len(timing_data["wf"]) > 0 and np.mean(timing_data["wf"]) <= 0.5
            efhr_success = len(timing_data["efhr"]) > 0 and 0.8 <= np.mean(timing_data["efhr"]) <= 1.2
            
            print(f"\n🎯 WfWorker timing: {'✅ PASS' if wf_success else '❌ FAIL'}")
            print(f"📊 EfHrWorker timing: {'✅ PASS' if efhr_success else '❌ FAIL'}")
            
            if wf_success and efhr_success:
                print("\n🎉 All timing requirements met!")
            else:
                print("\n⚠️  Some timing requirements not met")
                    
    except ConnectionRefusedError:
        print("❌ Cannot connect to server!")
        print("Make sure the server is running: uvicorn app:app --reload")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_timing())
