#!/usr/bin/env python3
"""
Jetson Workstation Attention Tracking System
Integrates YOLOv8 via TensorRT, SORT Multi-Object Tracking, and Dynamic Table Tracking
to generate statistics of occupancy and visitor engagement.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import tensorrt as trt
import torchvision

# Import self-contained modules
from utils.general import letterbox, scale_coords
from sort.sort import Sort

# Default configurations
DEFAULT_CONFIG = {
    "width": 1280,
    "height": 720,
    "stream_fps": 10,
    "yolo_interval": 5,
    "img_size": 320,
    "engine_path": "/data/yolov8n.engine",
    "rtsp_out": "rtsp://127.0.0.1:8554/live",
    "person_class": 0,
    "table_class": 60,
    "max_table_misses": 150,
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45
}


def load_config(config_path="configs/config.json"):
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
                config.update(file_config)
                print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Warning: Failed to load {config_path} ({e}). Using defaults.")
    else:
        print(f"Configuration file {config_path} not found. Using defaults.")
    return config


def get_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('8.8.8.8', 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def compute_iou(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    
    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


class DynamicTableTracker:
    def __init__(self, max_misses=150):
        self.tables = {}  # id -> dict
        self.next_id = 1
        self.max_misses = max_misses
        
    def update(self, detected_boxes, dt):
        matched_detection_indices = set()
        matched_table_ids = set()
        
        for tid, tbl in list(self.tables.items()):
            best_iou = 0.4  # IoU threshold to match
            best_idx = -1
            for idx, det_box in enumerate(detected_boxes):
                if idx in matched_detection_indices:
                    continue
                iou = compute_iou(tbl['bbox'], det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx != -1:
                # Smooth table box updates to reduce camera jitter
                tbl['bbox'] = (0.9 * np.array(tbl['bbox']) + 0.1 * np.array(detected_boxes[best_idx])).tolist()
                tbl['misses'] = 0
                tbl['hits'] += 1
                matched_detection_indices.add(best_idx)
                matched_table_ids.add(tid)
                
        for tid, tbl in list(self.tables.items()):
            if tid not in matched_table_ids:
                tbl['misses'] += 1
                if tbl['misses'] > self.max_misses:
                    del self.tables[tid]
                    
        for idx, det_box in enumerate(detected_boxes):
            if idx not in matched_detection_indices:
                self.tables[self.next_id] = {
                    'id': self.next_id,
                    'bbox': det_box,
                    'hits': 1,
                    'misses': 0,
                    'occupancy_sec': 0.0,
                    'occupied_last_frame': False,
                    'visitor_count': 0,
                    'visitor_ids': set(),
                    'current_visitors': set()
                }
                self.next_id += 1
                
    def process_person_tracks(self, tracks, dt):
        for tbl in self.tables.values():
            tbl['current_visitors'] = set()
            
        for trk in tracks:
            x1, y1, x2, y2, pid = trk
            pid = int(pid)
            
            # Person center coordinates
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            for tbl in self.tables.values():
                if tbl['hits'] < 2:
                    continue
                tx1, ty1, tx2, ty2 = tbl['bbox']
                
                # Check if person center overlaps with table box (with 10% padding)
                pad_w = (tx2 - tx1) * 0.1
                pad_h = (ty2 - ty1) * 0.1
                if (tx1 - pad_w <= cx <= tx2 + pad_w) and (ty1 - pad_h <= cy <= ty2 + pad_h):
                    tbl['current_visitors'].add(pid)
                    
        for tbl in self.tables.values():
            if tbl['hits'] < 2:
                continue
            if len(tbl['current_visitors']) > 0:
                tbl['occupancy_sec'] += dt
                tbl['occupied_last_frame'] = True
                for pid in tbl['current_visitors']:
                    if pid not in tbl['visitor_ids']:
                        tbl['visitor_ids'].add(pid)
                        tbl['visitor_count'] += 1
            else:
                tbl['occupied_last_frame'] = False
                
        stats = {}
        for tid, tbl in self.tables.items():
            if tbl['hits'] < 2:
                continue
            stats[tid] = {
                'id': tid,
                'bbox': tbl['bbox'],
                'occupancy_sec': tbl['occupancy_sec'],
                'visitor_count': tbl['visitor_count'],
                'occupied': tbl['occupied_last_frame']
            }
        return stats


class TensorRTYOLO:
    def __init__(self, engine_path, device, img_size=320, conf_threshold=0.25, iou_threshold=0.45):
        self.device = device
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        self.names = [''] * 80
        self.names[0] = 'person'
        self.names[60] = 'dining table'
        
        self.output_shape = tuple(self.engine.get_binding_shape(1))
        self.output_tensor = torch.empty(self.output_shape, dtype=torch.float32, device=self.device)

    def detect(self, frame):
        img_t = preprocess_for_yolo(frame, self.device, self.img_size)
        img_t = img_t.float()
        
        bindings = [img_t.data_ptr(), self.output_tensor.data_ptr()]
        self.context.execute_v2(bindings)
        
        output = self.output_tensor[0].t()
        
        boxes = output[:, 0:4]
        scores = output[:, 4:84]
        
        max_scores, class_ids = torch.max(scores, dim=1)
        
        keep = max_scores > self.conf_threshold
        
        if not keep.any():
            return np.empty((0, 6), dtype=np.float32)
            
        boxes = boxes[keep]
        scores = max_scores[keep]
        class_ids = class_ids[keep]
        
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_corners = torch.stack((x1, y1, x2, y2), dim=1)
        
        nms_keep = torchvision.ops.nms(boxes_corners, scores, iou_threshold=self.iou_threshold)
        
        if len(nms_keep) == 0:
            return np.empty((0, 6), dtype=np.float32)
            
        boxes_corners = boxes_corners[nms_keep]
        scores = scores[nms_keep]
        class_ids = class_ids[nms_keep]
        
        boxes_corners = scale_coords(
            (self.img_size, self.img_size),
            boxes_corners,
            frame.shape
        ).round()
        
        detections = torch.cat((boxes_corners, scores.unsqueeze(1), class_ids.unsqueeze(1).float()), dim=1)
        return detections.cpu().numpy().astype(np.float32, copy=False)


def preprocess_for_yolo(frame, device, img_size=320):
    img = letterbox(frame, img_size, stride=32, auto=False)[0]
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).to(device)
    img_t = img_t.half() / 255.0
    return img_t.unsqueeze(0)


def ffmpeg_input_args(w, h, fps):
    return [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pixel_format', 'bgr24',
        '-video_size', f'{w}x{h}',
        '-framerate', str(fps),
        '-i', 'pipe:0',
    ]


def ffmpeg_output_args(rtsp_url):
    return [
        '-rtsp_transport', 'tcp',
        '-f', 'rtsp',
        rtsp_url,
    ]


def _make_env():
    import os
    env = os.environ.copy()
    tegra_paths = '/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra-egl'
    if 'LD_LIBRARY_PATH' in env:
        env['LD_LIBRARY_PATH'] = tegra_paths + ':' + env['LD_LIBRARY_PATH']
    else:
        env['LD_LIBRARY_PATH'] = tegra_paths
    return env


def _try_gstreamer_hw(env, w, h, fps, rtsp_url):
    try:
        subprocess.check_output(
            ['gst-inspect-1.0', 'omxh264enc'],
            stderr=subprocess.DEVNULL,
            env=env,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        print("GStreamer omxh264enc: not found, skipping hardware acceleration")
        return None

    gst_cmd = [
        'gst-launch-1.0', '-q',
        'fdsrc', f'blocksize={w * h * 3}', 'do-timestamp=true',
        '!', f'video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1',
        '!', 'videoconvert',
        '!', 'video/x-raw,format=BGRx',
        '!', 'nvvidconv',
        '!', 'video/x-raw(memory:NVMM),format=NV12',
        '!', 'omxh264enc', 'insert-sps-pps=true', 'insert-vui=true',
        '!', 'video/x-h264,stream-format=byte-stream',
        '!', 'h264parse',
        '!', 'fdsink',
    ]

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'h264',
        '-i', 'pipe:0',
        '-c:v', 'copy',
        '-rtsp_transport', 'tcp',
        '-f', 'rtsp',
        rtsp_url,
    ]

    try:
        gst_proc = subprocess.Popen(
            gst_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=gst_proc.stdout,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        gst_proc.stdout.close()
    except OSError as e:
        print(f"GStreamer omxh264enc: launch failed: {e}")
        return None

    time.sleep(1.0)

    if gst_proc.poll() is not None or ffmpeg_proc.poll() is not None:
        for p in [gst_proc, ffmpeg_proc]:
            if p.poll() is None:
                p.kill()
        print("GStreamer omxh264enc: pipeline failed to start")
        return None

    try:
        gst_proc.stdin.write(bytearray(w * h * 3))
        gst_proc.stdin.flush()
        time.sleep(0.5)
    except IOError:
        for p in [gst_proc, ffmpeg_proc]:
            if p.poll() is None:
                p.kill()
        print("GStreamer omxh264enc: test write failed")
        return None

    if gst_proc.poll() is None and ffmpeg_proc.poll() is None:
        print("Encoder: omxh264enc via GStreamer (hardware, active and verified)")
        return gst_proc

    for p in [gst_proc, ffmpeg_proc]:
        if p.poll() is None:
            p.kill()
    print("GStreamer omxh264enc: pipeline died after test")
    return None


def encoder_candidates():
    available_encoders = []
    try:
        output = subprocess.check_output(
            ['ffmpeg', '-encoders'],
            stderr=subprocess.DEVNULL,
        ).decode('utf-8')
        for line in output.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith('V'):
                available_encoders.append(parts[1])
    except Exception as e:
        print(f"Failed to query FFmpeg encoders: {e}")

    candidates = []
    for enc in ['h264_nvmpi', 'h264_omx', 'h264_v4l2m2m']:
        if enc in available_encoders:
            candidates.append((enc, ['-c:v', enc, '-pix_fmt', 'yuv420p']))

    candidates.append(('libx264', [
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p',
    ]))
    return candidates


def start_ffmpeg(w, h, fps, rtsp_url):
    env = _make_env()
    base_args = ffmpeg_input_args(w, h, fps)
    output_args = ffmpeg_output_args(rtsp_url)

    gst_proc = _try_gstreamer_hw(env, w, h, fps, rtsp_url)
    if gst_proc is not None:
        return gst_proc

    chosen_name = None
    chosen_args = None

    for encoder_name, encoder_args in encoder_candidates():
        cmd = base_args + encoder_args + output_args
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        time.sleep(0.5)

        alive = proc.poll() is None
        if alive:
            try:
                proc.stdin.write(bytearray(w * h * 3))
                proc.stdin.flush()
                time.sleep(0.2)
            except IOError:
                alive = False
            alive = proc.poll() is None

        if alive:
            chosen_name = encoder_name
            chosen_args = encoder_args
            proc.stdin.close()
            proc.kill()
            proc.wait()
            break

        err_msg = ""
        try:
            if proc.stderr:
                err_msg = proc.stderr.read().decode('utf-8', errors='ignore')
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()

        print(f"FFmpeg encoder unavailable: {encoder_name}")
        for line in err_msg.splitlines():
            low = line.lower()
            if any(k in low for k in ['error', 'not found', 'could not', 'failed', 'can\'t']):
                print(f"  {line.strip()}")

    if chosen_name is None:
        raise RuntimeError("No FFmpeg H.264 encoder started")

    prod_proc = subprocess.Popen(
        base_args + chosen_args + output_args,
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    time.sleep(0.3)
    if prod_proc.poll() is not None:
        raise RuntimeError(f"FFmpeg encoder '{chosen_name}' crashed on re-launch")

    print(f"FFmpeg encoder: {chosen_name} (active)")
    return prod_proc


def camera_command(w, h, fps):
    return (
        'gst-launch-1.0 -q nvarguscamerasrc ! '
        f'video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! '
        'nvvidconv ! video/x-raw,format=BGRx ! '
        'videoconvert ! video/x-raw,format=BGR ! '
        'fdsink'
    )


def start_camera(w, h, fps):
    return subprocess.Popen(
        camera_command(w, h, fps).split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def make_frame_buffer(w, h):
    frame_size = w * h * 3
    raw_buffer = bytearray(frame_size)
    raw_view = memoryview(raw_buffer)
    frame = np.frombuffer(raw_buffer, dtype=np.uint8).reshape((h, w, 3))
    return raw_view, frame


def read_exact(stream, buffer_view):
    total = 0
    size = len(buffer_view)
    while total < size:
        bytes_read = stream.readinto(buffer_view[total:])
        if not bytes_read:
            return False
        total += bytes_read
    return True


def draw_tracks(frame, tracks, track_classes, class_names):
    colors = [
        (0, 255, 255), (255, 128, 0), (0, 255, 0), (0, 165, 255),
        (255, 0, 255), (255, 255, 0), (128, 0, 128), (0, 128, 128)
    ]
    for trk in tracks:
        x1, y1, x2, y2, tid = trk.astype(int)
        
        cls_id = track_classes.get(tid, 0)
        cls_name = class_names[cls_id] if hasattr(class_names, '__getitem__') and cls_id < len(class_names) else "Person"
        color = colors[cls_id % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{cls_name} {tid}",
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def draw_stats_overlay(frame, stats):
    num_tables = len(stats)
    if num_tables == 0:
        return
        
    box_height = 80 + num_tables * 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (910, 10), (1270, 10 + box_height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    green = (0, 255, 0)
    
    cv2.putText(frame, "WORKSTATION ATTENTION", (930, 35), font, 0.5, yellow, 1, cv2.LINE_AA)
    cv2.putText(frame, "Table ID", (925, 65), font, 0.45, yellow, 1, cv2.LINE_AA)
    cv2.putText(frame, "Occupied", (1030, 65), font, 0.45, yellow, 1, cv2.LINE_AA)
    cv2.putText(frame, "Visitors", (1160, 65), font, 0.45, yellow, 1, cv2.LINE_AA)
    cv2.line(frame, (925, 75), (1255, 75), (150, 150, 150), 1)
    
    y_offset = 100
    for tid in sorted(stats.keys()):
        tbl = stats[tid]
        occ_min = tbl['occupancy_sec'] / 60.0
        visitors = tbl['visitor_count']
        
        label_color = green if tbl['occupied'] else white
        
        cv2.putText(frame, f"Table {tid}", (925, y_offset), font, 0.45, label_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{occ_min:.1f} min", (1030, y_offset), font, 0.45, white, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{visitors}", (1160, y_offset), font, 0.45, white, 1, cv2.LINE_AA)
        y_offset += 30


def save_summary_report(stats, filepath="table_attention_summary.json"):
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            str(tid): {
                "occupancy_minutes": round(tbl["occupancy_sec"] / 60.0, 2),
                "total_visitors": tbl["visitor_count"]
            } for tid, tbl in stats.items()
        }
    }
    try:
        with open(filepath, "w") as f:
            json.dump(report, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save summary report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Jetson Workstation Attention Tracking")
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to configuration JSON')
    parser.add_argument('--engine', type=str, help='Path to TensorRT engine file (overrides config)')
    parser.add_argument('--rtsp-out', type=str, help='Output RTSP URL (overrides config)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Overrides
    if args.engine:
        config["engine_path"] = args.engine
    if args.rtsp_out:
        config["rtsp_out"] = args.rtsp_out

    w = config["width"]
    h = config["height"]
    fps = config["stream_fps"]
    yolo_interval = config["yolo_interval"]
    img_size = config["img_size"]
    engine_path = config["engine_path"]
    rtsp_out = config["rtsp_out"]
    person_class = config["person_class"]
    table_class = config["table_class"]
    max_table_misses = config["max_table_misses"]
    conf_threshold = config["confidence_threshold"]
    iou_threshold = config["iou_threshold"]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize detector
    if not os.path.exists(engine_path):
        print(f"Error: TensorRT engine not found at {engine_path}")
        sys.exit(1)
        
    model = TensorRTYOLO(
        engine_path=engine_path,
        device=device,
        img_size=img_size,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )
    tx1_ip = get_ip()

    print("Model loaded successfully.")
    print(f"Device IP: {tx1_ip}")
    print(f"Output Stream: {rtsp_out}")
    print(f"YOLO Processing Interval: {yolo_interval} frames")

    raw_view, frame = make_frame_buffer(w, h)
    in_proc = start_camera(w, h, fps)

    print("Warming up camera...")
    for _ in range(3):
        if not read_exact(in_proc.stdout, raw_view):
            raise RuntimeError("Camera failed during warmup")
    print("Camera ready.")

    out_proc = start_ffmpeg(w, h, fps, rtsp_out)

    mot_tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)
    track_classes = {}
    table_tracker = DynamicTableTracker(max_misses=max_table_misses)

    frame_count = 0
    empty_dets = np.empty((0, 6), dtype=np.float32)
    last_dets = empty_dets
    start_time = time.time()
    last_time = time.time()
    last_report_time = time.time()

    try:
        while True:
            if not read_exact(in_proc.stdout, raw_view):
                print("Camera read failed")
                break

            current_time = time.time()
            dt = min(current_time - last_time, 1.0)
            last_time = current_time

            # Perform inference at intervals
            if frame_count % yolo_interval == 0:
                last_dets = model.detect(frame)
                
                # Filter person detections for SORT
                person_dets = last_dets[last_dets[:, 5] == person_class]
                if len(person_dets) > 0:
                    tracks = mot_tracker.update(person_dets[:, :5])
                else:
                    tracks = mot_tracker.update(np.empty((0, 5)))
                    
                for trk in tracks:
                    tid = int(trk[4])
                    track_classes[tid] = person_class
                    
                # Filter table detections
                table_dets = last_dets[last_dets[:, 5] == table_class]
                table_boxes = table_dets[:, :4].tolist()
                table_tracker.update(table_boxes, dt)
            else:
                mot_tracker.frame_count += 1
                predicted_tracks = []
                for trk in mot_tracker.trackers:
                    pos = trk.predict()[0]
                    if not np.any(np.isnan(pos)):
                        if trk.hits >= mot_tracker.min_hits and trk.time_since_update <= mot_tracker.max_age:
                            predicted_tracks.append(np.concatenate((pos, [trk.id + 1])))
                if len(predicted_tracks) > 0:
                    tracks = np.stack(predicted_tracks)
                else:
                    tracks = np.empty((0, 5))
                    
            # Process track coordinates & dynamic ROIs
            stats = table_tracker.process_person_tracks(tracks, dt)

            # Draw HUD stats overlay
            draw_stats_overlay(frame, stats)

            # Draw table bounding boxes
            for tid, tbl in stats.items():
                bx1, by1, bx2, by2 = [int(v) for v in tbl['bbox']]
                color = (0, 255, 0) if tbl['occupied'] else (0, 165, 255)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
                cv2.putText(frame, f"Table {tid}", (bx1 + 10, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw person tracks
            draw_tracks(frame, tracks, track_classes, model.names)

            # Write raw frame to output encoder pipe
            try:
                out_proc.stdin.write(raw_view)
            except BrokenPipeError:
                print("FFmpeg output pipe broke")
                break

            frame_count += 1

            # Log Performance & Periodic Reports
            if frame_count % 100 == 0:
                fps_rate = frame_count / (time.time() - start_time)
                print(f"FPS: {fps_rate:.2f}")
                
            # Periodically write json report every 30 seconds
            if current_time - last_report_time > 30.0:
                save_summary_report(stats, "table_attention_summary.json")
                last_report_time = current_time

    except KeyboardInterrupt:
        print("Stopping tracking script...")
    finally:
        # Cleanup
        in_proc.kill()
        out_proc.kill()
        save_summary_report(stats, "table_attention_summary.json")
        print("Cleanup completed.")


if __name__ == '__main__':
    main()
