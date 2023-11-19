import cv2
import torch
import os
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate, prioritize_outermost_largest
from ultralytics import YOLO

# Thiết lập codec cho VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'H264')

# Load models
vehicle_detector = YOLO("./models/vehicle_detector.pt").to("cpu")
license_plate_detector = YOLO("./models/license_plate_detector.pt").to("cpu")

# Load video
cap = cv2.VideoCapture("./videos/night-sight-002.mp4")

# Kích thước khung hình của video đầu vào
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Mở VideoWriter với codec và tham số khác
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (frame_width, frame_height))

# Other setup
vehicles = [2, 3]
frame_skip_interval = 2
frame_number = -1
mot_tracker = Sort()
license_plate_counts = {}

while True:
    frame_number += 1
    ret, frame = cap.read()
    if not ret:
        print("No return....")
        break

    if frame_number % frame_skip_interval != 0:
        continue

    frame_tensor = torch.from_numpy(frame).to("cpu")
    detections = vehicle_detector(frame)[0].to("cpu")
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score, int(class_id)])

    # Prioritize outermost and largest bounding box
    detections_ = prioritize_outermost_largest(detections_)

    for box in detections_:
        x1, y1, x2, y2, _, class_id = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Hiển thị loại phương tiện lên trên bounding box
        class_name = "Unknown"
        if class_id == 2:
            class_name = "Car"
        elif class_id == 3:
            class_name = "Motorcycle"

        # Tăng kích thước chữ
        font_size = 1.0
        font_thickness = 2
        cv2.putText(frame, class_name, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), font_thickness)

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates
    license_plates = license_plate_detector(frame)[0].to("cpu")

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, _, _ = license_plate
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        # Crop license plate image
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        vehicle_crop = frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2), :]

        if xcar1 < xcar2 and ycar1 < ycar2:
            vehicle_crop = frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2), :]

            if vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
                cv2.imshow("vehicle_crop", vehicle_crop)
        
        # Vẽ khung xung quanh biển số
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Process license plate image
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV
        )

        cv2.imshow("origin_crop", license_plate_crop)
        cv2.imshow("threshold_crop", license_plate_crop_thresh)

        # Read license plate using EasyOCR
        license_plate_text, _ = read_license_plate(license_plate_crop_thresh)
        if license_plate_text is not None:
            license_plate_counts[license_plate_text] = (
                license_plate_counts.get(license_plate_text, 0) + 1
            )

            current_occurrences = license_plate_counts[license_plate_text]

            if current_occurrences > 1:
                last_frame_filename = (
                    f"result_images/{license_plate_text}_{current_occurrences - 1}.png"
                )

                if os.path.exists(last_frame_filename):
                    os.remove(last_frame_filename)
                    print(f"Deleted old frame: {last_frame_filename}")
            
            # Ghi thông tin biển số lên trên bounding box
            font_size = 1.0
            font_thickness = 2
            cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thickness)

            print("--->> License plate:", license_plate_text)
            print("      Number of occurrences:", license_plate_counts[license_plate_text])

            frame_filename = f"result_images/{license_plate_text}_{license_plate_counts[license_plate_text]}.png"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame as {frame_filename}")

    out.write(frame)

    # Show video (reduced resolution for display)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Detected Video", small_frame)

    if cv2.waitKey(30) == 27:
        print("Esc...")
        break

    
cap.release()
out.release()
cv2.destroyAllWindows()
