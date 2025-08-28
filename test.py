#!/usr/bin/env python3
"""
test.py - Simple script to test WebSocket timing

This script starts the server and runs the timing test automatically.
"""

import subprocess
import asyncio
import time
import signal
import sys
import os

def main():
    print("🚀 WebSocket Timing Test")
    print("=" * 50)
    print("Testing WfWorker (≤0.5s) vs EfHrWorker (~1s) timing...")
    print("=" * 50)
    
    server_process = None
    
    try:
        # Start server
        print("📡 Starting server...")
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "app:app", 
            "--host", "127.0.0.1", "--port", "8000"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        print("⏳ Waiting for server startup...")
        time.sleep(5)
        
        # Run test
        print("🧪 Running timing test...\n")
        result = subprocess.run([sys.executable, "timing_test.py"])
        
        if result.returncode == 0:
            print("\n✅ Test completed successfully!")
        else:
            print("\n❌ Test encountered errors")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Clean up server
        if server_process:
            print("🛑 Stopping server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
