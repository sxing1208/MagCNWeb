# MagCNWeb

## Introduction

MagCNWeb is a comprehensive software package developed by the Bai Group at the University of North Carolina at Chapel Hill. It enables magnetic field decoding, cardiac contractility (ejection fraction) computation, and real-time cardiac rhythm classification for an ESP32-S3–based, Wi-Fi–enabled Hall-effect sensor array used in conjunction with cardiac magnetic implants.

The system streams a 64-channel, 12-bit ADC output that is reconstructed into an 8 × 8 matrix for downstream processing.

---

## File Description

- utils.py: Barebones implementation of the core algorithm (PAGO-DeMag) for magnetic field decoding and reconstruction  
- utils_fast.py: Performance-enhanced version of utils.py leveraging parallel processing  
- workers_sync.py: Worker threads for parallel decoding and classification  
- app.py: Main FastAPI application for web interface and device communication  
- dummy.py: Simulated external device for streaming test data  

---

## Requirements

The software package was tested on:

- Windows 11  
- Python 3.11  

Required Python packages (tested with versions available as of August 2025):

- numpy  
- scipy  
- tensorflow  
- fastapi  
- uvicorn  
- websockets  
- opencv-python  

All packages are available via PyPI.

---

## Installation

Install git by using [winget] in Powershell
[winget install --id Git.Git -e --source winget]

Install Python 3.11 from the official website:  
[https://www.python.org/downloads/release/python-3110/]

Clone the repository:

```bash
git clone https://github.com/sxing1208/MagCNWeb  
cd MagCNWeb
```

Install dependencies:

```bash
py -V:3.11 -m pip install opencv-python numpy matplotlib websockets tensorflow uvicorn fastapi scipy
```

Typical installation time is approximately 10 minutes, including dependencies.

---

## Demo and Usage

### Local Demo

Run the application:
```bash
py -V:3.11 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

In a separate terminal, run the simulated device:
```bash
py -V:3.11 dummy.py
```

Open a browser and navigate to:

[http://127.0.0.1:8000]

Expected output:
- Reduced calculated ejection fraction  
- Heart rate around 100 bpm  
- Normal sinus rhythm classification  
- Real-time waveform visualization  

---

## Using a Wi-Fi Peripheral (AP Mode)

The following instructions assume usage of the Bai Group peripheral configured in Wi-Fi AP mode, streaming 64-channel data via WebSockets.

### Step 1: Connect to Device Network
Connect your computer to the Wi-Fi network emitted by the peripheral.

### Step 2: Start Server (Network Accessible)
```bash
py -V:3.11 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Step 3: View on Host Computer

[http://127.0.0.1:8000]

---

## Mobile Visualization

1. Connect your mobile device to the same network (ESP32 AP)

2. On the host computer, find the local IP address:
```bash
ipconfig
```
Example:
```
IPv4 Address . . . . . . . . . . : 192.168.4.2
```
3. On your mobile device, open:

[http://192.168.4.2:8000] (Or your custom IP address)

---

## License

This repository is released under the MIT License.
