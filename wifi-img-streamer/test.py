#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2

# --- TRACKBAR SETUP FOR ANY COLOR ---
def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.createTrackbar('H_Min', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('S_Min', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('V_Min', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('H_Max', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S_Max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V_Max', 'Trackbars', 255, 255, nothing)


# --- TASK 1 & 2 LOGIC ---
def process_frame(frame):
    # ==========================================
    # TASK 1: Edge Detection
    # ==========================================
    edges = cv2.Canny(frame, 100, 200)

    # ==========================================
    # TASK 2: ANY Color Tracking
    # ==========================================
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos('H_Min', 'Trackbars')
    s_min = cv2.getTrackbarPos('S_Min', 'Trackbars')
    v_min = cv2.getTrackbarPos('V_Min', 'Trackbars')
    h_max = cv2.getTrackbarPos('H_Max', 'Trackbars')
    s_max = cv2.getTrackbarPos('S_Max', 'Trackbars')
    v_max = cv2.getTrackbarPos('V_Max', 'Trackbars')

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)
    mask = cv2.erode(mask, None, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    direction = "None"

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area > 200:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.drawMarker(frame, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

            height, width = frame.shape[:2]
            img_center_x, img_center_y = width // 2, height // 2

            dx = cx - img_center_x
            dy = cy - img_center_y

            TARGET_AREA = 5000
            da = TARGET_AREA - area

            abs_dx = abs(dx)
            abs_dy = abs(dy)
            abs_da = abs(da) * 0.02
            max_dev = max(abs_dx, abs_dy, abs_da)

            if max_dev == abs_dx:
                direction = "Right" if dx > 0 else "Left"
            elif max_dev == abs_dy:
                direction = "Down" if dy > 0 else "Up"
            else:
                direction = "Front" if da > 0 else "Back"

    # ==========================================
    # TASK 3: Circle Detection
    # ==========================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circle_detected = False
    raw_circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                                    param1=100, param2=60, minRadius=40, maxRadius=300)
    if raw_circles is not None:
        raw_circles = np.uint16(np.around(raw_circles))
        for c in raw_circles[0, :]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            # Verify it's a solid continuous blob using a circular mask
            mask_circle = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask_circle, (cx, cy), r, 255, -1)
            roi = cv2.bitwise_and(gray_blur, gray_blur, mask=mask_circle)
            filled_pixels = np.count_nonzero(roi)
            total_pixels = np.count_nonzero(mask_circle)
            fill_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
            if fill_ratio > 0.5:  # at least 50% filled = continuous blob
                cv2.circle(frame, (cx, cy), r, (0, 255, 0), 3)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                circle_detected = True

    if circle_detected:
        cv2.putText(frame, "Circle", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    print(f"Intended Flight Command: {direction}")
    cv2.putText(frame, f"Cmd: {direction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return frame, edges, mask, direction, circle_detected


# --- WEBCAM ---
print("Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame.")
        break

    processed_frame, edges_frame, mask_frame, command, saw_circle = process_frame(frame)

    cv2.imshow('Task 2: Tracking & Bounding Box', processed_frame)
    cv2.imshow('Task 1: Canny Edges', edges_frame)
    cv2.imshow('Debug: Color Mask', mask_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
