# Jetson Workstation Attention Tracking System

This repository contains a professional, self-contained implementation of an automated office workstation attention tracking system deployed on a NVIDIA Jetson TX1 edge computer.

The system captures a camera stream, performs high-speed object detection via **TensorRT-accelerated YOLOv8**, tracks people trajectories using the **SORT (Simple Online and Realtime Tracking)** algorithm, and dynamically monitors workstation (table) occupancy and visitor count.

---

## 🚀 Key Features

* **Hardware-Accelerated Inference**: Uses custom FP16 TensorRT bindings for YOLOv8 (running at **~110 FPS raw throughput** on Jetson TX1).
* **Zero-Copy Memory Access**: Feeds PyTorch GPU tensors directly into the TensorRT engine context, completely bypassing host-to-device memory copies.
* **SORT Tracker**: High-efficiency multi-object tracking (using Kalman filters and Hungarian data association) to track people trajectories.
* **Dynamic Workstation ROI Tracking**: Detects tables (class 60) and dynamically tracks their coordinates, applying a temporal filter to smooth box coordinates and handling short-term camera adjustments or drift.
* **Visitor and Occupancy Metrics**: Calculates active workstation occupancy duration (in minutes) and tracks the count of unique visitors entering the workstation zones.
* **RTSP Streaming HUD Overlay**: Renders a glassmorphism statistics overlay HUD in the top-right corner, highlights workstations with dynamic color-coding (green for occupied, orange for empty), and broadcasts the stream to a RTSP server.

---

## 📂 Directory Structure

```
jetson-workstation-tracker/
├── configs/
│   ├── config.json       # Main JSON parameters configuration
│   └── mediamtx.yml      # MediaMTX RTSP server configurations
├── sort/
│   ├── __init__.py
│   └── sort.py           # Self-contained SORT Kalman filter tracking module
├── utils/
│   ├── __init__.py
│   └── general.py        # OpenCV letterboxing & coordinate scaling helpers
├── .gitignore            # Git exclusions (ignores weights, caches, and logs)
├── README.md             # This setup and documentation guide
├── requirements.txt      # Python dependencies manifest
└── yolo_stream_sort_v8.py # Main workstation tracking pipeline script
```

---

## 🛠️ Prerequisites & Installation

### 1. Hardware Requirements
* NVIDIA Jetson TX1 / TX2 / Nano / Xavier (Running JetPack 4.x or 5.x)
* CSI camera or USB web-camera (GStreamer-compatible)

### 2. System Packages
Install system-level encoders and media libraries:
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools
```

### 3. Model Compilation (NVIDIA trtexec)
On your host or Jetson, compile the YOLOv8 nano model into a native TensorRT engine to run with FP16 precision:

1. **Export YOLOv8 to ONNX** (on host machine):
   ```python
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.export(format="onnx", imgsz=320, opset=12)
   ```
2. **Compile ONNX on Jetson** via `trtexec`:
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=yolov8n.onnx --saveEngine=/data/yolov8n.engine --fp16
   ```

### 4. Setup Python Environment
Create a virtual environment on the Jetson and install Python dependencies:
```bash
python3 -m venv cleanenv
source cleanenv/bin/activate
pip install -r requirements.txt
```
> **Note:** PyTorch and Torchvision should be installed using the official NVIDIA pre-compiled wheels for JetPack.

---

## ⚙️ Configuration (`configs/config.json`)

All configurations are handled inside [configs/config.json](configs/config.json). You can modify resolution, frame rates, tracking intervals, thresholds, and paths:

* `width` / `height`: Resolution of camera capture (default `1280x720`).
* `stream_fps`: Capture frame rate (default `10`).
* `yolo_interval`: Perform deep learning inference every N frames (interpolation handles intermediate frames).
* `img_size`: YOLOv8 input shape dimension (default `320`).
* `engine_path`: Path to the compiled TensorRT `.engine` file.
* `rtsp_out`: Endpoint where the processed stream will be published.
* `confidence_threshold` / `iou_threshold`: YOLO model confidence and NMS overlap thresholds.

---

## 🏃 Running the Project

### 1. Run the RTSP Server
Download [MediaMTX](https://github.com/bluenviron/mediamtx) on the Jetson, place it in the project path or `/data/`, and run it:
```bash
# From the project root
./mediamtx configs/mediamtx.yml > mediamtx.log 2>&1 &
```

### 2. Launch the Tracking Pipeline
Run the main tracking script using the compiled virtual environment:
```bash
# Activate your environment
source cleanenv/bin/activate

# Launch the script (will automatically read configs/config.json)
python yolo_stream_sort_v8.py

# Optional overrides via command line:
# python yolo_stream_sort_v8.py --engine /custom/path/yolov8n.engine --rtsp-out rtsp://127.0.0.1:8554/custom_live
```

### 3. View the Stream
You can view the live stream from any computer on the same local network using VLC or FFplay:
```bash
ffplay rtsp://<JETSON_IP_ADDRESS>:8554/live
```

---

## 📊 Output Metrics Schema
Every 30 seconds, the active statistics are saved to a file named `table_attention_summary.json` with the following structure:
```json
{
    "timestamp": "2026-07-13 16:05:32",
    "data": {
        "1": {
            "occupancy_minutes": 14.5,
            "total_visitors": 4
        },
        "2": {
            "occupancy_minutes": 45.2,
            "total_visitors": 12
        }
    }
}
```

---

## 🛡️ License & Copyright
This project uses the SORT tracker which is licensed under the GPL-3.0 License. See the `sort/sort.py` file for license headers.
