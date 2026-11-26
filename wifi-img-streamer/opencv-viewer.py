#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS378 Assignment 3 — Crazyflie Vision-Based Tracking

Uses AI deck camera over WiFi for vision, radio for flight control.

Keybinds (focus any OpenCV window first):
  T — toggle tracking mode ON / OFF
        ON  → drone follows the red circle
        OFF → drone holds position (no motion commands sent)
  Q — quit and land

Usage:
  Preview only:   python opencv-viewer.py
  With flight:    python opencv-viewer.py --fly
"""
import argparse
import logging
import socket
import struct
import threading
import time

import cv2
import numpy as np

import libusb_package
import usb.backend.libusb1
_b = libusb_package.get_libusb1_backend()
usb.backend.libusb1.get_backend = lambda find_library=None: _b

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

logging.basicConfig(level=logging.ERROR)

# --- CONFIG ---
RADIO_URI = 'radio://0/80/2M/E7E7E7E7E7'
AIDECK_IP = '192.168.4.1'
AIDECK_PORT = 5000

# --- Frame throttle ---
# Only process 1 in every N frames for motion decisions.
# Display still updates every frame so the feed looks live.
PROCESS_EVERY_N_FRAMES = 1

# --- Motion tuning ---
MOVE_DISTANCE    = 0.12   # meters per lateral / depth step
MOVE_HEIGHT      = 0.08   # meters per up/down step
COMMAND_INTERVAL = 2    # min seconds between motion commands
CONFIRM_FRAMES   = 3      # direction must be stable this many processed frames

# --- Vision tuning ---
TARGET_RADIUS   = 50
CENTER_DEADZONE = 25
DEPTH_DEADZONE  = 12
EMA_ALPHA       = 0.35
LOCK_LOST_FRAMES = 3      # keep last target this many processed frames after a brief miss
LOCKED_AREA_SCALE = 0.65  # while locked, accept a slightly smaller contour before dropping
MIN_CONTOUR_AREA_FLOOR = 20
MIN_HOUGH_RADIUS_FLOOR = 3

RED_LOWER1 = np.array([0,    70,  35])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([160,  70,  35])
RED_UPPER2 = np.array([179, 255, 255])

# --- Trackbars ---
def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar('Use_Fixed_Red',   'Trackbars', 1,    1,   nothing)
cv2.createTrackbar('H_Min',           'Trackbars', 0,    179, nothing)
cv2.createTrackbar('S_Min',           'Trackbars', 120,  255, nothing)
cv2.createTrackbar('V_Min',           'Trackbars', 60,   255, nothing)
cv2.createTrackbar('H_Max',           'Trackbars', 10,   179, nothing)
cv2.createTrackbar('S_Max',           'Trackbars', 255,  255, nothing)
cv2.createTrackbar('V_Max',           'Trackbars', 255,  255, nothing)
cv2.createTrackbar('Min_Area',        'Trackbars', 80,   5000, nothing)
cv2.createTrackbar('Min_Radius',      'Trackbars', 4,    100, nothing)
cv2.createTrackbar('Edge_TargetLuma', 'Trackbars', 95,   180, nothing)
cv2.createTrackbar('Edge_Floor',      'Trackbars', 18,   100, nothing)

CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# --- EMA smoothing ---
_smooth = {'cx': None, 'cy': None, 'r': None}
_lost_lock_frames = 0

def _ema(prev, new):
    if prev is None:
        return float(new)
    return EMA_ALPHA * float(new) + (1.0 - EMA_ALPHA) * prev


# ---------------------------------------------------------------------------
# Motion controller — discrete steps with confirmation
# ---------------------------------------------------------------------------
class MotionController:
    """
    Issues motion commands only when tracking mode is enabled.
    Requires the same direction for CONFIRM_FRAMES consecutive processed
    frames, then fires one discrete MotionCommander step.
    In hover mode all propose() calls are silently ignored.
    """

    def __init__(self, mc: MotionCommander):
        self.mc             = mc
        self._lock          = threading.Lock()
        self._busy          = False
        self._last_cmd_time = 0.0
        self._pending_dir   = "None"
        self._confirm_count = 0
        self.tracking       = False

    def toggle_tracking(self):
        self.tracking = not self.tracking
        self._pending_dir   = "None"
        self._confirm_count = 0
        state = "ON  — drone will follow circle" if self.tracking else "OFF — drone hovering"
        print(f"[TRACKING] {state}")

    def propose(self, direction: str):
        if not self.tracking:
            return

        if direction == self._pending_dir:
            self._confirm_count += 1
        else:
            self._pending_dir   = direction
            self._confirm_count = 1

        if self._confirm_count < CONFIRM_FRAMES:
            return

        now = time.time()
        if now - self._last_cmd_time < COMMAND_INTERVAL:
            return

        with self._lock:
            if self._busy:
                return
            self._busy = True

        self._last_cmd_time = now
        threading.Thread(target=self._run, args=(direction,), daemon=True).start()

    def _run(self, direction: str):
        try:
            self._dispatch(direction)
        finally:
            with self._lock:
                self._busy = False

    def _dispatch(self, direction: str):
        mc = self.mc
        print(f"[MOVE] {direction}")
        if   direction == "Left":   mc.left(MOVE_DISTANCE)
        elif direction == "Right":  mc.right(MOVE_DISTANCE)
        elif direction == "Up":     mc.up(MOVE_HEIGHT)
        elif direction == "Down":   mc.down(MOVE_HEIGHT)
        elif direction == "Front":  mc.forward(MOVE_DISTANCE)
        elif direction == "Back":   mc.back(MOVE_DISTANCE)


# ---------------------------------------------------------------------------
# AI deck camera stream (WiFi TCP + CPX)
# ---------------------------------------------------------------------------
def rx_bytes(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed by AI-deck.")
        data.extend(chunk)
    return data


class AIDeckCamera:
    def __init__(self, deck_ip, deck_port):
        self.deck_ip   = deck_ip
        self.deck_port = deck_port
        self._sock     = None

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.deck_ip, self.deck_port))
        print(f"AI deck camera connected at {self.deck_ip}:{self.deck_port}")

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def read_frame(self):
        if not self._sock:
            raise RuntimeError("Not connected.")
        raw          = rx_bytes(self._sock, 4)
        length, _, _ = struct.unpack('<HBB', raw)
        hdr          = rx_bytes(self._sock, length - 2)
        magic, width, height, depth, img_fmt, size = struct.unpack('<BHHBBI', hdr)
        if magic != 0xBC:
            return None
        stream = bytearray()
        while len(stream) < size:
            raw2       = rx_bytes(self._sock, 4)
            clen, _, _ = struct.unpack('<HBB', raw2)
            stream.extend(rx_bytes(self._sock, clen - 2))
        if img_fmt == 0:
            arr = np.frombuffer(stream, dtype=np.uint8)
            if arr.size < width * height * max(depth, 1):
                return None
            bayer = arr[:width * height].reshape((height, width))
            return cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)
        return cv2.imdecode(np.frombuffer(stream, np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Low-light edge enhancement
# ---------------------------------------------------------------------------
def prepare_low_light_edges(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, _, _ = cv2.split(lab)

    target_luma  = max(40, cv2.getTrackbarPos('Edge_TargetLuma', 'Trackbars'))
    current_luma = float(np.mean(l_channel))
    if current_luma < target_luma:
        alpha     = min(3.0, max(1.0, target_luma / max(current_luma, 1.0)))
        beta      = min(80,  int((target_luma - current_luma) * 0.5))
        l_channel = cv2.convertScaleAbs(l_channel, alpha=alpha, beta=beta)

    l_channel     = CLAHE.apply(l_channel)
    enhanced_gray = cv2.GaussianBlur(l_channel, (5, 5), 0)

    median_v   = float(np.median(enhanced_gray))
    edge_floor = max(5, cv2.getTrackbarPos('Edge_Floor', 'Trackbars'))
    edge_low   = int(max(edge_floor, 0.66 * median_v))
    edge_high  = int(min(255, max(edge_low + 25, 1.45 * median_v + 10)))

    return enhanced_gray, edge_low, edge_high


# ---------------------------------------------------------------------------
# Color mask
# ---------------------------------------------------------------------------
def build_color_mask(hsv_frame):
    use_fixed = cv2.getTrackbarPos('Use_Fixed_Red', 'Trackbars')
    if use_fixed:
        m1   = cv2.inRange(hsv_frame, RED_LOWER1, RED_UPPER1)
        m2   = cv2.inRange(hsv_frame, RED_LOWER2, RED_UPPER2)
        mask = cv2.bitwise_or(m1, m2)
    else:
        lower = np.array([cv2.getTrackbarPos('H_Min', 'Trackbars'),
                          cv2.getTrackbarPos('S_Min', 'Trackbars'),
                          cv2.getTrackbarPos('V_Min', 'Trackbars')])
        upper = np.array([cv2.getTrackbarPos('H_Max', 'Trackbars'),
                          cv2.getTrackbarPos('S_Max', 'Trackbars'),
                          cv2.getTrackbarPos('V_Max', 'Trackbars')])
        mask = cv2.inRange(hsv_frame, lower, upper)
    mask = cv2.erode(mask,  None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


# ---------------------------------------------------------------------------
# Direction logic — single-axis priority
# ---------------------------------------------------------------------------
def compute_direction(cx, cy, radius, frame_w, frame_h):
    dx = cx - frame_w // 2
    dy = cy - frame_h // 2
    dr = radius - TARGET_RADIUS

    eff_dx = abs(dx) if abs(dx) > CENTER_DEADZONE else 0
    eff_dy = abs(dy) if abs(dy) > CENTER_DEADZONE else 0
    eff_dr = abs(dr) if abs(dr) > DEPTH_DEADZONE  else 0

    if eff_dx == 0 and eff_dy == 0 and eff_dr == 0:
        return "Hover", dx, dy, dr
    if eff_dx >= eff_dy and eff_dx >= eff_dr:
        return ("Right" if dx > 0 else "Left"),  dx, dy, dr
    elif eff_dy >= eff_dx and eff_dy >= eff_dr:
        return ("Down"  if dy > 0 else "Up"),    dx, dy, dr
    else:
        return ("Back"  if dr > 0 else "Front"), dx, dy, dr


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------
def process_frame(frame, tracking_active: bool):
    global _smooth, _lost_lock_frames
    raw_frame = frame.copy()
    enhanced_gray, edge_low, edge_high = prepare_low_light_edges(raw_frame)
    frame_h, frame_w = frame.shape[:2]

    # Task 1: Edge Detection
    edges = cv2.Canny(enhanced_gray, edge_low, edge_high)

    # Color mask
    hsv  = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)
    mask = build_color_mask(hsv)

    # Task 2: Bounding box
    min_area = max(MIN_CONTOUR_AREA_FLOOR, cv2.getTrackbarPos('Min_Area', 'Trackbars'))
    area_gate = min_area
    if _smooth['cx'] is not None:
        area_gate = max(MIN_CONTOUR_AREA_FLOOR, int(min_area * LOCKED_AREA_SCALE))
    color_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    color_cx = color_cy = color_radius = None
    if color_contours:
        c    = max(color_contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > area_gate:
            x, y, w, h  = cv2.boundingRect(c)
            color_cx     = x + w // 2
            color_cy     = y + h // 2
            color_radius = int((w + h) / 4)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.drawMarker(frame, (color_cx, color_cy), (0, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

    # Task 3: Circle detection on color mask
    best_circle = None
    detection_source = "None"
    if color_cx is not None:
        mask_blur = cv2.GaussianBlur(mask, (9, 9), 2)
        min_dim   = min(frame_h, frame_w)
        min_dist  = max(20, int(min_dim * 0.15))
        min_r_user = max(MIN_HOUGH_RADIUS_FLOOR, cv2.getTrackbarPos('Min_Radius', 'Trackbars'))
        min_r     = max(min_r_user, color_radius // 3)
        max_r     = max(min_r + 5, int(min_dim * 0.48))

        raw_circles = cv2.HoughCircles(
            mask_blur, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min_dist, param1=50, param2=15,
            minRadius=min_r, maxRadius=max_r,
        )
        if raw_circles is not None:
            circles   = np.uint16(np.around(raw_circles[0, :]))
            best_dist = float('inf')
            for c in circles:
                cxc, cyc, rc = int(c[0]), int(c[1]), int(c[2])
                dist = (cxc - color_cx)**2 + (cyc - color_cy)**2
                if dist < best_dist:
                    best_dist   = dist
                    best_circle = (cxc, cyc, rc)
                    detection_source = "ColorHough"

    if best_circle is None and color_cx is not None:
        best_circle = (color_cx, color_cy, color_radius)
        detection_source = "ColorBox"

    # Fallback: if color misses in low light, try circles in enhanced grayscale.
    if best_circle is None:
        gray_blur = cv2.GaussianBlur(enhanced_gray, (9, 9), 2)
        min_dim   = min(frame_h, frame_w)
        min_r     = max(MIN_HOUGH_RADIUS_FLOOR, cv2.getTrackbarPos('Min_Radius', 'Trackbars'))
        max_r     = max(min_r + 8, int(min_dim * 0.48))
        gray_circles = cv2.HoughCircles(
            gray_blur, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(30, int(min_dim * 0.2)),
            param1=70, param2=18,
            minRadius=min_r, maxRadius=max_r,
        )
        if gray_circles is not None:
            circles = np.around(gray_circles[0, :]).astype(int)
            if _smooth['cx'] is not None and _smooth['cy'] is not None:
                ref_x = int(_smooth['cx'])
                ref_y = int(_smooth['cy'])
                best_dist = float('inf')
                for c in circles:
                    cxc, cyc, rc = int(c[0]), int(c[1]), int(c[2])
                    dist = (cxc - ref_x)**2 + (cyc - ref_y)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_circle = (cxc, cyc, rc)
            else:
                c = min(circles, key=lambda v: int(v[2]))
                best_circle = (int(c[0]), int(c[1]), int(c[2]))
            if best_circle is not None:
                detection_source = "GrayHough"

    # EMA smoothing
    fresh_detection = best_circle is not None
    detected_cx = detected_cy = detected_r = None
    if best_circle is not None:
        _lost_lock_frames = 0
        raw_cx, raw_cy, raw_r = best_circle
        _smooth['cx'] = _ema(_smooth['cx'], raw_cx)
        _smooth['cy'] = _ema(_smooth['cy'], raw_cy)
        _smooth['r']  = _ema(_smooth['r'],  raw_r)
        detected_cx = int(_smooth['cx'])
        detected_cy = int(_smooth['cy'])
        detected_r  = int(_smooth['r'])
        cv2.circle(frame, (detected_cx, detected_cy), detected_r, (0, 255, 0), 3)
        cv2.circle(frame, (detected_cx, detected_cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"r={detected_r}px",
                    (detected_cx + detected_r + 4, detected_cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        if _smooth['cx'] is not None and _lost_lock_frames < LOCK_LOST_FRAMES:
            _lost_lock_frames += 1
            detected_cx = int(_smooth['cx'])
            detected_cy = int(_smooth['cy'])
            detected_r  = int(_smooth['r'])
            cv2.circle(frame, (detected_cx, detected_cy), detected_r, (0, 180, 255), 2)
            cv2.circle(frame, (detected_cx, detected_cy), 4, (0, 180, 255), -1)
            cv2.putText(frame, f"hold:{_lost_lock_frames}",
                        (detected_cx + detected_r + 4, detected_cy + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
        else:
            _smooth = {'cx': None, 'cy': None, 'r': None}
            _lost_lock_frames = 0

    # Direction
    direction = "None"
    dx = dy = dr = 0
    if detected_r is not None:
        direction, dx, dy, dr = compute_direction(
            detected_cx, detected_cy, detected_r, frame_w, frame_h)

    # Frame-centre crosshair
    cv2.drawMarker(frame, (frame_w // 2, frame_h // 2), (255, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

    # HUD
    banner_color = (0, 220, 0) if tracking_active else (0, 0, 220)
    banner_text  = "TRACKING: ON  (T to stop)" if tracking_active else "TRACKING: OFF (T to start)"
    cv2.rectangle(frame, (0, 0), (frame_w, 28), (0, 0, 0), -1)
    cv2.putText(frame, banner_text,
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2)

    cv2.putText(frame, f"Cmd: {direction}",
                (20, 55),  cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (255, 255,   0), 2)
    cv2.putText(frame, f"dx:{dx:+d} dy:{dy:+d} dr:{dr:+d}",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1)
    cv2.putText(frame, f"Edge:{edge_low}/{edge_high}",
                (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    cv2.putText(frame, f"R:{TARGET_RADIUS}px CZ:{CENTER_DEADZONE} DZ:{DEPTH_DEADZONE}",
                (20, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)
    use_fixed = cv2.getTrackbarPos('Use_Fixed_Red', 'Trackbars')
    cv2.putText(frame, f"Color: {'FixedRed' if use_fixed else 'Trackbar'}  "
                       f"Skip:{PROCESS_EVERY_N_FRAMES}f",
                (20, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
    cv2.putText(frame, f"Src:{detection_source}",
                (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

    print(f"[{'TRACK' if tracking_active else 'HOVER'}] Cmd:{direction} "
          f"dx={dx:+d} dy={dy:+d} dr={dr:+d}")

    return frame, edges, mask, direction, fresh_detection, enhanced_gray


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Crazyflie vision tracking — Assignment 3')
    p.add_argument('-n', '--deck-ip',   default=AIDECK_IP)
    p.add_argument('-p', '--deck-port', type=int, default=AIDECK_PORT)
    p.add_argument('--fly', action='store_true',
                   help='Enable flight mode (connects to Crazyflie via radio)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
# Cache last processed views for skipped frames
_last = [None]


def _run_loop(cam: AIDeckCamera, motion_ctrl, auto_track: bool = True):
    print("Keybinds: T = toggle tracking   Q = quit")
    frame_count = 0
    camera_tracking = bool(auto_track)

    if motion_ctrl is not None:
        motion_ctrl.tracking = bool(auto_track)

    if auto_track:
        state = "ON - drone will follow circle" if motion_ctrl is not None else "ON - visual tracking active"
        print(f"[TRACKING] {state}")

    while True:
        try:
            frame = cam.read_frame()
        except (ConnectionError, OSError, RuntimeError, struct.error, ValueError) as e:
            print(f"Frame error: {e}")
            break
        if frame is None:
            continue

        frame_count += 1
        tracking_active = motion_ctrl.tracking if motion_ctrl else camera_tracking

        # Only process every Nth frame for motion decisions
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            processed, edges, mask, direction, saw_circle, gray_dbg = \
                process_frame(frame, tracking_active)

            if motion_ctrl is not None:
                motion_ctrl.propose(direction if saw_circle else "Hover")

            _last[0] = (processed, edges, mask, gray_dbg)
        else:
            # On skipped frames: just overlay tracking banner on raw frame
            h, w = frame.shape[:2]
            banner_color = (0, 220, 0) if tracking_active else (0, 0, 220)
            banner_text  = ("TRACKING: ON  (T to stop)"
                            if tracking_active else "TRACKING: OFF (T to start)")
            cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
            cv2.putText(frame, banner_text,
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, banner_color, 2)
            if _last[0] is not None:
                _, edges, mask, gray_dbg = _last[0]
            else:
                edges = gray_dbg = mask = frame
            processed = frame

        cv2.imshow('Task 2: Tracking & Bounding Box', processed)
        cv2.imshow('Task 1: Canny Edges',             edges)
        cv2.imshow('Debug: Color Mask',               mask)
        cv2.imshow('Task 3: Circle Input',            gray_dbg)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('t'), ord('T')):
            if motion_ctrl is not None:
                motion_ctrl.toggle_tracking()
            else:
                camera_tracking = not camera_tracking
                state = "ON - visual tracking active" if camera_tracking else "OFF - visual tracking paused"
                print(f"[TRACKING] {state}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    cam = AIDeckCamera(args.deck_ip, args.deck_port)
    print(f"Connecting to AI-deck camera on {args.deck_ip}:{args.deck_port} ...")
    try:
        cam.connect()
    except OSError as e:
        print(f"Camera connection failed: {e}")
        return
    print("Camera connected.")

    if True:
        print(f"Connecting to Crazyflie at {RADIO_URI} ...")
        cflib.crtp.init_drivers()
        with SyncCrazyflie(RADIO_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            with MotionCommander(scf, default_height=0.4) as mc:
                print("Airborne. Auto-tracking is ON (press T to pause/resume).")
                _run_loop(cam, MotionController(mc), auto_track=True)
        print("Landed and disconnected.")
    else:
        print("Camera-only mode. Auto-tracking is ON. Pass --fly to enable flight.")
        _run_loop(cam, None, auto_track=True)

    cam.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    main()
