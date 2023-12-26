import cv2
import torch
import os
import numpy as np
from datetime import datetime
from sort.sort import Sort
from util import get_car, read_license_plate, prioritize_outermost_largest
from ultralytics import YOLO

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available. Using CPU.")

num_gpus = torch.cuda.device_count()

torch.cuda.set_device(0)

current_gpu = torch.cuda.current_device()
print(f"Using GPU-{current_gpu}")


# Thiết lập codec cho VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"H264")

# Load models
vehicle_detector = YOLO("./models/vehicle_detector.pt").to("cuda")
license_plate_detector = YOLO("./models/license_plate_detector.pt").to("cuda")

# Setting Lines and Variables
PINK_LINE = [(440, 840), (1920, 840)]
YELLOW_LINE = [(430, 860), (1920, 860)]
ORANGE_LINE = [(420, 900), (1920, 900)]

PREV_LINE = [(650, 150), (1920, 150)]
PASS_LINE = [(600, 840), (1920, 840)]

cross_pink_line = {}
cross_yellow_line = {}
cross_orange_line = {}

avg_speeds = {}

VIDEO_FPS = 88
FACTOR_KM = 3.6
LATENCY_FPS = 75


# Euclidean Distance Function
def euclidean_distance(point1: tuple, point2: tuple):
    x1, y1 = point1
    x2, y2 = point2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance


# Speed Calculation
def calculate_avg_speed(track_id):
    time_bg = (
        cross_yellow_line[track_id]["time"] - cross_orange_line[track_id]["time"]
    ).total_seconds()
    time_gr = (
        cross_pink_line[track_id]["time"] - cross_yellow_line[track_id]["time"]
    ).total_seconds()

    distance_bg = euclidean_distance(
        cross_yellow_line[track_id]["point"], cross_orange_line[track_id]["point"]
    )
    distance_gr = euclidean_distance(
        cross_pink_line[track_id]["point"], cross_yellow_line[track_id]["point"]
    )

    speed_bg = round(
        (distance_bg / (time_bg * VIDEO_FPS)) * (FACTOR_KM * LATENCY_FPS), 2
    )
    speed_gr = round(
        (distance_gr / (time_gr * VIDEO_FPS)) * (FACTOR_KM * LATENCY_FPS), 2
    )

    return round((speed_bg + speed_gr) / 2, 2)


# Load video
cap = cv2.VideoCapture("./videos/day_sight_1.mp4")

# Kích thước khung hình của video đầu vào
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Mở VideoWriter với codec và tham số khác
out = cv2.VideoWriter(
    "./output_videos/output_video.mp4", fourcc, 20.0, (frame_width, frame_height)
)

# Other setup
vehicles = [2, 3]
frame_skip_interval = 2
frame_number = -1
prev_time = datetime.now()
mot_tracker = Sort()
license_plate_counts = {}

while True:
    frame_number += 1
    ret, frame = cap.read()

    # Calculate FPS
    current_time = datetime.now()
    elapsed_time = (current_time - prev_time).total_seconds()
    fps = 1 / elapsed_time
    prev_time = current_time

    # Display FPS on the frame
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    height = frame.shape[0]
    width = frame.shape[1]

    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array([[[360, 1080], [700, 0], [1920, 0], [1920, 1080]]])
    pts.reshape((-1, 1, 2))
    cv2.fillPoly(mask, pts, 255)

    zone = cv2.bitwise_and(frame, frame, mask=mask)

    # Hiển thị vùng quan tâm
    green_overlay = frame.copy()
    cv2.fillPoly(green_overlay, pts, (0, 255, 0))  # Màu xanh lục
    cv2.addWeighted(green_overlay, 0.2, frame, 0.8, 0, frame)  # Opacity 0.2

    # Vẽ các đường dùng để xác định tốc độ
    cv2.line(frame, PASS_LINE[0], PASS_LINE[1], (255, 255, 255), 3)
    cv2.line(frame, PREV_LINE[0], PREV_LINE[1], (255, 255, 255), 3)
    cv2.line(frame, PINK_LINE[0], PINK_LINE[1], (255, 0, 255), 3)  # Đường màu hồng
    cv2.line(frame, YELLOW_LINE[0], YELLOW_LINE[1], (0, 255, 255), 3)  # Đường màu vàng
    cv2.line(frame, ORANGE_LINE[0], ORANGE_LINE[1], (0, 165, 255), 3)  # Đường màu cam

    if not ret:
        print("No return....")
        break

    if frame_number % frame_skip_interval != 0:
        continue

    frame_tensor = torch.from_numpy(zone).to("cuda")
    detections = vehicle_detector(zone)[0].to("cpu")
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score, int(class_id)])

    # Xác định bounding box lớn nhất của phương tiện đó
    detections_ = prioritize_outermost_largest(detections_)

    for box in detections_:
        x1, y1, x2, y2, _, class_id = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Xác định và hiển thị loại phương tiện lên trên bounding box
        class_name = "Unknown"
        if class_id == 2:
            class_name = "Car"
        elif class_id == 3:
            class_name = "Motorcycle"

        cv2.putText(
            frame,
            class_name,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

    # Theo dõi phương tiện
    track_ids = mot_tracker.update(np.asarray(detections_))
    track_ids = track_ids.astype(int)
    print("Track_ids: ", track_ids)

    # Xác định tốc độ phương tiện
    for xmin, ymin, xmax, ymax, track_id in track_ids:
        xc, yc = int((xmin + xmax) / 2), ymax

        if track_id not in cross_orange_line:
            cross_orange = (ORANGE_LINE[1][0] - ORANGE_LINE[0][0]) * (
                yc - ORANGE_LINE[0][1]
            ) - (ORANGE_LINE[1][1] - ORANGE_LINE[0][1]) * (xc - ORANGE_LINE[0][0])
            if cross_orange <= 0:
                cross_orange_line[track_id] = {
                    "time": datetime.now(),
                    "point": (xc, yc),
                }
        elif track_id not in cross_yellow_line and track_id in cross_orange_line:
            cross_yellow = (YELLOW_LINE[1][0] - YELLOW_LINE[0][0]) * (
                yc - YELLOW_LINE[0][1]
            ) - (YELLOW_LINE[1][1] - YELLOW_LINE[0][1]) * (xc - YELLOW_LINE[0][0])
            if cross_yellow <= 0:
                cross_yellow_line[track_id] = {
                    "time": datetime.now(),
                    "point": (xc, yc),
                }

        elif track_id not in cross_pink_line and track_id in cross_yellow_line:
            cross_pink = (PINK_LINE[1][0] - PINK_LINE[0][0]) * (
                yc - PINK_LINE[0][1]
            ) - (PINK_LINE[1][1] - PINK_LINE[0][1]) * (xc - PINK_LINE[0][0])
            if cross_pink <= 0:
                cross_pink_line[track_id] = {
                    "time": datetime.now(),
                    "point": (xc, yc),
                }

                avg_speed = calculate_avg_speed(track_id)
                avg_speeds[track_id] = f"{avg_speed} km/h"

        cross_line = (PREV_LINE[1][0] - PREV_LINE[0][0]) * (yc - PREV_LINE[0][1]) - (
            PREV_LINE[1][1] - PREV_LINE[0][1]
        ) * (xc - PREV_LINE[0][0])

        if (
            track_id in avg_speeds
            and track_id in cross_orange_line
            and track_id in cross_yellow_line
            and track_id in cross_pink_line
            and cross_line >= 0
        ):
            cv2.putText(
                frame,
                avg_speeds[track_id],
                (int(xmin), int(ymin) - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            

    # Xác định biển số phương tiện
    license_plates = license_plate_detector(zone)[0].to("cpu")

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, _, _ = license_plate
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        # Crop ảnh biển số phương tiện
        license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

        # Vẽ khung xung quanh biển số
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Tiền xử lí hình ảnh biển số
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV
        )

        cv2.imshow("origin_crop", license_plate_crop)
        cv2.imshow("threshold_crop", license_plate_crop_thresh)

        # Đọc biển số sử dụng EasyOCR
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
                    # print(f"Deleted old frame: {last_frame_filename}")

            # Ghi kết quả đọc biển số lên trên bounding box
            cv2.putText(
                frame,
                license_plate_text,
                (int(x1), int(y1) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

            # print("--->> License plate:", license_plate_text)
            # print(
            #     "      Number of occurrences:", license_plate_counts[license_plate_text]
            # )

            frame_filename = f"result_images/{license_plate_text}_{license_plate_counts[license_plate_text]}.png"
            cv2.imwrite(frame_filename, frame)
            # print(f"Saved frame as {frame_filename}")

    out.write(frame)

    # Show video và giảm độ phân giải
    small_frame = cv2.resize(frame, (1067, 600))
    cv2.imshow("Detected Video", small_frame)

    if cv2.waitKey(30) == 27:
        print("Esc...")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
