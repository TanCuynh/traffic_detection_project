from ultralytics import YOLO
import cv2

array_result = []
import torch

if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available. Using CPU.")

num_gpus = torch.cuda.device_count()

torch.cuda.set_device(0)

current_gpu = torch.cuda.current_device()
print(f"Using GPU-{current_gpu}")

from sort.sort import *
from util import get_car, read_license_plate

cv2.namedWindow("Detected Video", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detected Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

results = {}

mot_tracker = Sort()

# load models
vehicle_detector = YOLO("./models/vehicle_detector.pt").to("cuda")
license_plate_detector = YOLO("./models/license_plate_detector.pt").to("cuda")

# load video
cap = cv2.VideoCapture("./videos/night-sight-002.mp4")

vehicles = [2, 3]

# Khoảng cách giữa các frame bạn muốn xử lý
frame_skip_interval = 2  # Xử lý mỗi 2 frame

frame_number = -1

# Thêm biến để lưu trữ số lần xuất hiện của mỗi biển số
license_plate_counts = {}
last_detected_license_plate = None
max_occurrences = 0

import os

while True:
    frame_number += 1  # Tăng biến frame
    ret, frame = cap.read()
    if not ret:
        print("No return....")
        break

    # Bỏ qua frame nếu không phải frame cần xử lý
    if frame_number % frame_skip_interval != 0:
        continue

    print("Running....")
    frame_tensor = torch.from_numpy(frame).to("cuda")
    results[frame_number] = {}
    detections = vehicle_detector(frame)[0].to("cpu")
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Vẽ bounding boxes xung quanh các đối tượng đã phát hiện
    for box in detections_:
        x1, y1, x2, y2, _ = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Hiển thị video trong chế độ full màn hình
    cv2.imshow("Detected Video", frame)

    if detections_:
        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

    # áp dụng model phát hiện biển số xe
    license_plates = license_plate_detector(frame)[0].to("cpu")

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        # crop hình ảnh biển số
        license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]
        vehicle_crop = frame[int(ycar1) : int(ycar2), int(xcar1) : int(xcar2), :]

        # Kiểm tra valid bounding box coordinates
        if xcar1 < xcar2 and ycar1 < ycar2:
            # crop hình ảnh phương tiện
            vehicle_crop = frame[int(ycar1) : int(ycar2), int(xcar1) : int(xcar2), :]

            # Kiểm tra valid cropped image
            if vehicle_crop.shape[0] > 0 and vehicle_crop.shape[1] > 0:
                # Hiển thị hình ảnh của phương tiện
                cv2.imshow("vehicle_crop", vehicle_crop)

        # xử lí hình ảnh biển số được phát hiện
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray, 100, 255, cv2.THRESH_BINARY_INV
        )

        cv2.imshow("origin_crop", license_plate_crop)
        cv2.imshow("threshold_crop", license_plate_crop_thresh)

        # đọc biển số sử dụng EasyOCR
        license_plate_text, license_plate_text_score = read_license_plate(
            license_plate_crop_thresh
        )
        if license_plate_text is not None:
            # Tăng số lần xuất hiện của biển số trong từ điển
            license_plate_counts[license_plate_text] = (
                license_plate_counts.get(license_plate_text, 0) + 1
            )

            # Lấy số lần xuất hiện của biển số
            current_occurrences = license_plate_counts[license_plate_text]

            # Kiểm tra và xóa hình ảnh cũ nếu đã xuất hiện trước đó
            if current_occurrences > 1:
                # Lấy tên file của ảnh cũ
                last_frame_filename = (
                    f"result_images/{license_plate_text}_{current_occurrences - 1}.png"
                )

                # Kiểm tra xem tập tin có tồn tại không trước khi xóa
                if os.path.exists(last_frame_filename):
                    # Xóa tập tin cũ
                    os.remove(last_frame_filename)
                    print(f"Deleted old frame: {last_frame_filename}")

            # In thông tin và số lần xuất hiện
            print("--->> License plate:", license_plate_text)
            print("      Possibility:", license_plate_text_score)
            print(
                "      Number of occurrences:", license_plate_counts[license_plate_text]
            )

            # Lưu frame với tên file chứa số lần xuất hiện
            frame_filename = f"result_images/{license_plate_text}_{license_plate_counts[license_plate_text]}.png"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame as {frame_filename}")

    if cv2.waitKey(30) == 27:
        print("Esc...")
        break
