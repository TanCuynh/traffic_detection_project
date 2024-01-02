import cv2
import torch
import os
import numpy as np
from datetime import datetime
from sort.sort import Sort
from util import get_car, read_license_plate, prioritize_outermost_largest
from ultralytics import YOLO
from flask import Flask, render_template, Response, request, send_from_directory
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, resources={r"/": {"origins": ""}})


@app.route("/reports/<path:path>")
def send_report(path):
    return send_from_directory("static", path)


def gen():
    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("GPU is not available. Using CPU.")

    num_gpus = torch.cuda.device_count()

    torch.cuda.set_device(0)

    current_gpu = torch.cuda.current_device()
    print(f"Using GPU-{current_gpu}")

    fourcc = cv2.VideoWriter_fourcc(*"H264")

    vehicle_detector = YOLO("./models/vehicle_detector.pt").to("cuda")
    license_plate_detector = YOLO("./models/license_plate_detector.pt").to("cuda")

    # Xác định biến và đường xác định tốc độ
    PINK_LINE = [(520, 600), (1920, 600)]
    YELLOW_LINE = [(470, 750), (1920, 750)]
    ORANGE_LINE = [(420, 900), (1920, 900)]

    PREV_LINE = [(650, 150), (1920, 150)]
    PASS_LINE = [(600, 600), (1920, 600)]

    cross_pink_line = {}
    cross_yellow_line = {}
    cross_orange_line = {}

    avg_speeds = {}

    kmh_to_ms_const = 3.6
    pixel_to_meter_const = 0.1

    def euclidean_distance(point1: tuple, point2: tuple):
        x1, y1 = point1
        x2, y2 = point2
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distance

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
            ((distance_bg * kmh_to_ms_const * pixel_to_meter_const * 0.5) / time_bg), 2
        )
        speed_gr = round(
            ((distance_gr * kmh_to_ms_const * pixel_to_meter_const * 0.5) / time_gr), 2
        )

        return round((speed_bg + speed_gr) / 2, 2)

    cap = cv2.VideoCapture("./videos/day_sight_1.mp4")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(
        "./static/output_videos/output_video.mp4",
        fourcc,
        20.0,
        (frame_width, frame_height),
    )

    frame_skip_interval = 2
    frame_number = -1
    mot_tracker = Sort()
    license_plate_counts = {}

    while True:
        frame_number += 1
        ret, frame = cap.read()

        height = frame.shape[0]
        width = frame.shape[1]

        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array([[[360, 1080], [700, 0], [1920, 0], [1920, 1080]]])
        pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, pts, 255)

        zone = cv2.bitwise_and(frame, frame, mask=mask)

        # green_overlay = frame.copy()
        # cv2.fillPoly(green_overlay, pts, (0, 255, 0))
        # cv2.addWeighted(green_overlay, 0.2, frame, 0.8, 0, frame)

        # cv2.line(frame, PASS_LINE[0], PASS_LINE[1], (255, 255, 255), 3)
        # cv2.line(frame, PREV_LINE[0], PREV_LINE[1], (255, 255, 255), 3)
        # cv2.line(frame, PINK_LINE[0], PINK_LINE[1], (255, 0, 255), 3)
        # cv2.line(frame, YELLOW_LINE[0], YELLOW_LINE[1], (0, 255, 255), 3)
        # cv2.line(frame, ORANGE_LINE[0], ORANGE_LINE[1], (0, 165, 255), 3)

        if not ret:
            print("No return....")
            break

        if frame_number % frame_skip_interval != 0:
            continue

        frame_tensor = torch.from_numpy(zone).to("cuda")

        # Xác định phương tiện
        detections = vehicle_detector(zone, classes=[2, 3])[0].to("cpu")
        detections_ = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            detections_.append([x1, y1, x2, y2, score, int(class_id)])

        detections_ = prioritize_outermost_largest(detections_)

        class_name = "Unknown"

        for box in detections_:
            x1, y1, x2, y2, _, class_id = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

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

        track_ids = mot_tracker.update(np.asarray(detections_))
        track_ids = track_ids.astype(int)

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
                    avg_speeds[track_id] = f"{avg_speed}kmh"

            cross_line = (PREV_LINE[1][0] - PREV_LINE[0][0]) * (
                yc - PREV_LINE[0][1]
            ) - (PREV_LINE[1][1] - PREV_LINE[0][1]) * (xc - PREV_LINE[0][0])

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
                    (255, 255, 0),
                    2,
                )

        # Xác định biển số phương tiện
        license_plates = license_plate_detector(zone)[0].to("cpu")
        exist_image = []

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, _, _ = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # Crop ảnh biển số phương tiện
            license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

            # Vẽ khung xung quanh biển số
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # Tiền xử lí hình ảnh biển số
            license_plate_crop_gray = cv2.cvtColor(
                license_plate_crop, cv2.COLOR_BGR2GRAY
            )

            # resized_license_plate_crop = cv2.resize(license_plate_crop, (300, 300))
            # cv2.imshow("Original License", resized_license_plate_crop)
            # resized_license_plate_crop_gray = cv2.resize(
            #     license_plate_crop_gray, (300, 300)
            # )
            # cv2.imshow("Pre-processed License", resized_license_plate_crop_gray)

            # Đọc biển số
            license_plate_text, _ = read_license_plate(license_plate_crop_gray)

            if license_plate_text is not None:
                cv2.putText(
                    frame,
                    license_plate_text,
                    (int(x1), int(y1) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

                current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                license_filename = ""

                if car_id in avg_speeds:
                    license_filename = f"{car_id}-{class_name}-{avg_speeds[car_id]}-{license_plate_text}.png"
                elif car_id != -1:
                    license_filename = (
                        f"{car_id}-{class_name}-Not Detected-{license_plate_text}.png"
                    )
                if license_filename not in exist_image:
                    cv2.imwrite(
                        f"static/result_licenses/{current_time}-" + license_filename,
                        license_plate_crop,
                    )
                    cv2.imwrite(
                        f"static/result_frames/{current_time}-" + license_filename,
                        frame,
                    )
                    exist_image.append(license_filename)
                    print(f"Deleted old frame: {license_filename}")
        # out.write(frame)

        cv2.imwrite("demo.jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + open("demo.jpg", "rb").read()
            + b"\r\n"
        )

        # cv2.imshow("Detected Video", frame)

        if cv2.waitKey(30) == 27:
            print("Esc...")
            break


@app.route("/video_feed", methods=["GET", "POST"])
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=True)
