# MagCNWeb

## Introduction

**MagCNWeb** is a comprehensive software package developed by the Bai group at the University of North Carolina at Chapel Hill
that enables magnetic field decoding, cardiac contractility (ejection fraction) computation, and real-time cardiac rhythm classification
for ESP32S3-based, wifi enabled Hall-effect sensor array used in conjunction with cardiac magnetic implants.

## File Description

*utils.py*: A barebones implementation of the core algorithm (PAGO-DeMag) for magnetic field decoding and reconstruction

*utils_fast.py*: A performance enhanced version of utils.py that leverages parallel processing

*works_sync.py*: The Main worker threads that conducts parallel processing for decoding and classifcation

*app.py*: Main App for web peripheral handling

*dummy.py*: A simulated "dummy" external web device that streams simulated sensor data

## Requirements
The software package was tested on Windows 11 using python 3.11 and requires the latest version of numpy, websockets, scipy, tensorflow,
uvicron, cv2, and fastapi as of Aug 2025. All packages are available through PyPI and can be install through `pip install` command.
Additonally, standard python packages like logging, collections and typing are also required

## Intallation
To insall python 3.11, navigate to [this link](https://www.python.org/downloads/release/python-3110/) for an offical python installer:
The software package can be installed through cloning this github repository using the following command:
```bash
git clone https://github.com/sxing1208/MagCNWeb
```
all python packages can be installed through PyPI. The following bash command can be used for dependency installations.
```bash
cd MagCNWeb
py -V:3.11 -m pip install opencv-python numpy matplotlib websockets tensorflow uvicorn fastapi scipy
```
Typically, the installation process takes around 10min including source code and dependecies.

## Demo and Instruction for Usage
The user may choose to run a demo by running *app.py* locally along with *dummy.py*.
To begin the demo process, open a terminal in the same folder as MagCNWeb.
First, load app.py using unicorn using the following command
```bash
py -V:3.11 -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Then, in a separate terminal, run dummy.py
```bash
py -V:3.11 dummy.py
```
The user should expect to see a reduced calculated ejection fraction, a heart rate around 100 and a rhythm of normal sinus rhythm.

To connect to a wifi peripheral, ensure both devices are under the same network and make app.py available through local network

## License
This repo is available under MIT license.
