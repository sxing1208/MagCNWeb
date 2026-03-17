# MagCNWeb

## Introduction

**MagCNWeb** is a comprehensive software package developed by the Bai group at the University of North Carolina at Chapel Hill that enables magnetic field decoding, cardiac contractility (ejection fraction) computation, and real-time cardiac rhythm classification for an ESP32-S3–based, Wi-Fi–enabled Hall-effect sensor array used in conjunction with cardiac magnetic implants which streams a 64-channel, 12-bit ADC output that is reconstructed into an 8 × 8 matrix for downstream processing.

## File Description

*utils.py*: A barebones implementation of the core algorithm (PAGO-DeMag) for magnetic field decoding and reconstruction

*utils_fast.py*: A performance enhanced version of utils.py that leverages parallel processing

*workers_sync.py*: The Main worker threads that conducts parallel processing for decoding and classification

*app.py*: Main App for web peripheral handling

*dummy.py*: A simulated "dummy" external web device that streams simulated sensor data

## Requirements
The software package was tested on Windows 11 using python 3.11 and requires the latest version of numpy, websockets, scipy, tensorflow,
uvicorn, cv2, and fastapi available in Aug 2025. All packages are available through PyPI and can be installed through `pip install` command.
Additionally, standard python packages like logging, collections and typing are also required

## Installation
To intsall python 3.11, navigate to [this link](https://www.python.org/downloads/release/python-3110/) for an official python installer:
The software package can be installed through cloning this github repository using the following command:
```bash
git clone https://github.com/sxing1208/MagCNWeb
```
all python packages can be installed through PyPI. The following bash command can be used for dependency installations.
```bash
cd MagCNWeb
py -V:3.11 -m pip install opencv-python numpy matplotlib websockets tensorflow uvicorn fastapi scipy
```
Typically, the installation process takes around 10min including source code and dependencies.

## Demo and Instruction for Usage
The user may choose to run a demo by running *app.py* locally along with *dummy.py*.
To begin the demo process, open a terminal in the same folder as MagCNWeb.
First, load app.py using uvicorn using the following command
```bash
py -V:3.11 -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Then, in a separate terminal, run dummy.py
```bash
py -V:3.11 dummy.py
```
The user should expect to see a reduced calculated ejection fraction, a heart rate around 100 and a rhythm of normal sinus rhythm.
The user can also visualize the incoming waveform and continuous computational results by open [127.0.0.1:8000](127.0.0.1:8000) in the browser.

To connect to a wifi peripheral, follow the following step:

*The following instructions assumes usage of the peripheral developed by Bai Group at UNC that is configured in wifi AP mode
and streams 64 channel data through websockets*

Connect the computer to the wireless network emitted by the peripheral device.

Expose the local server to devices on the same network through the following command:
```bash
py -V:3.11 -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The peripheral device should stream the data through wifi now.
The user can also visualize the incoming waveform and continuous computational results in the host computer by open [127.0.0.1:8000](127.0.0.1:8000) in the browser.

To visualize the incoming waveform and continuous computational results on a mobile device, connect the mobile device to the network served by the peripheral device.

Then, check the the local device IP address of the server (computer) using `ipconfig`
```bash
ipconfig
```
Look for the IPv4 Address like this
```
IPv4 Address . . . . . . . . . . : 192.168.4.2
```
In the mobile device's browser, type the aforementioned IP address to access the datastream.

## License
This repo is available under MIT license.
