# MagCNWeb

## Introduction

**MagCNWeb** is a comprehensive software package developed by the Bai group at the University of North Carolina at Chapel Hill
that enables magnetic field decoding, cardiac contractility (ejection fraction) computation, and real-time cardiac rhythm classification
for ESP32S3-based, wifi enabled Hall-effect sensor array used in conjunction with cardiac magneitc implants.

## File Description

*utils.py*: A barebones implementation of the core algorithm (PAGO-DeMag) for magnetic field decoding and reconstruction
*utils_fast.py*: A performance enhanced version of utils.py that leverages parallel processing
*works_sync.py*: The Main worker threads that conducts parallel processing for decoding and classifcation
*app.py*: Main App for web peripheral handling
*dummy.py*: A simulated "dummy" external web device that streams simulated sensor data

## Requirements
The software package was tested on Windows 11 using python 3.11 and requires the latest version of numpy, websockets, scipy, tensorflow, logging
uvicron, and fastapi as of Aug 2025. All packages are available through PyPI and can be install through `pip install` command.
Additonally, standard python packages like collections and typing are also required

## Intallation
The software package can be installed through `git clone` all python packages can be installed through PyPI. Typicall, the installation
process takes around 10min including source code and dependecies.

## Demo and Instruction for Usage
The user may choose to run a demo by running "dummy.py."
After running, the user should load app.py using unicorn using the following command
```
uvicorn main:app --host 127.0.0.0 --port 8000
```
The user should expect to see a reduced calculated ejection fraction.

To connect to a wifi peripheral, ensure both devices are under the same network and make app.py available through local network

# License
This repo is available under MIT license.
