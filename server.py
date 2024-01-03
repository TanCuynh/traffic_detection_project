import cv2
import torch
import os
import numpy as np
from datetime import datetime
from sort.sort import Sort
from util import get_car, read_license_plate, prioritize_outermost_largest
from speed_detection_def import calculate_avg_speed
from ultralytics import YOLO
from flask import (
    Flask,
    render_template,
    Response,
    request,
    send_from_directory,
    jsonify,
)
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import sqlite3
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})


stop_streaming = False


@app.route("/clear_stream", methods=["POST"])
def clear_stream():
    global stop_streaming
    print("-------------------END DETECTING------------------")
    stop_streaming = True
    return jsonify({"message": "Streaming stopped successfully."})


def gen(value):
    global stop_streaming
    stop_streaming = False

    if torch.cuda.is_available():
        print("GPU is available.")
    else:
        print("GPU is not available. Using CPU.")

    torch.cuda.set_device(0)

    current_gpu = torch.cuda.current_device()
    print(f"Using GPU-{current_gpu}")

    fourcc = cv2.VideoWriter_fourcc(*"H264")

    vehicle_detector = YOLO("./models/vehicle_detector.pt").to("cuda")
    license_plate_detector = YOLO("./models/license_plate_detector.pt").to("cuda")

    PINK_LINE = [(520, 600), (1920, 600)]
    YELLOW_LINE = [(470, 750), (1920, 750)]
    ORANGE_LINE = [(420, 900), (1920, 900)]

    PREV_LINE = [(650, 150), (1920, 150)]
    PASS_LINE = [(600, 600), (1920, 600)]

    cross_pink_line = {}
    cross_yellow_line = {}
    cross_orange_line = {}

    avg_speeds = {}

    cap = cv2.VideoCapture("./videos/" + "day_sight_" + str(value) + ".mp4")

    frame_skip_interval = 2
    frame_number = -1
    mot_tracker = Sort()
    existed_images = []

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

                    avg_speed = calculate_avg_speed(
                        track_id, cross_yellow_line, cross_orange_line, cross_pink_line
                    )
                    avg_speeds[track_id] = avg_speed

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
                    f"{avg_speeds[track_id]} km/h",
                    (int(xmin), int(ymin) - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 0),
                    2,
                )

        # Xác định biển số phương tiện
        license_plates = license_plate_detector(zone)[0].to("cpu")

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, _, _ = license_plate
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            resized_license_plate_crop = cv2.resize(license_plate_crop, (250, 200))

            license_plate_crop_gray = cv2.cvtColor(
                license_plate_crop, cv2.COLOR_BGR2GRAY
            )

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

                if value == 1:
                    location = "Danang"
                elif value == 2:
                    location = "Hanoi"
                else:
                    location = "Ho Chi Minh City"

                if car_id in avg_speeds:
                    license_filename = f"Car, Mototcycle-{avg_speeds[car_id]}-{license_plate_text}-{location}.jpg"
                elif car_id != -1:
                    license_filename = f"Car, Mototcycle-Not Detected-{license_plate_text}-{location}.jpg"

                if license_filename not in existed_images:
                    cv2.imwrite(
                        f"static/result_licenses/{current_time}-" + license_filename,
                        resized_license_plate_crop,
                    )
                    cv2.imwrite(
                        f"static/result_frames/{current_time}-" + license_filename,
                        frame,
                    )
                    existed_images.append(license_filename)
                    print("EXISTED_IMAGES:", existed_images)

        cv2.imwrite("demo.jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + open("demo.jpg", "rb").read()
            + b"\r\n"
        )
        if stop_streaming:
            break

        if cv2.waitKey(30) == 27:
            print("Esc...")
            break


@app.route("/<int:value>/video_feed", methods=["GET", "POST"])
def video_feed(value):
    return Response(gen(value), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/push_into_db", methods=["POST", "GET"])
def push_into_db():
    def get_all_files_in_directory(directory):
        return os.listdir(directory)

    files_result_frames = get_all_files_in_directory("./static/result_frames")
    files_result_licenses = get_all_files_in_directory("./static/result_licenses")

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    with open("schema.sql") as f:
        conn.executescript(f.read())

    for file in files_result_frames:
        parts = file.split("-")
        time = parts[0]
        time_object = datetime.strptime(time, "%Y_%m_%d_%H_%M_%S")

        formatted_time = time_object.strftime("%d-%m-%Y %H:%M:%S")
        vehicle_type = parts[1]
        speed = parts[2]
        license_plate = parts[3]
        location = parts[4].replace(".jpg", "")

        result_file_name = file

        cur.execute(
            "SELECT * FROM vehicle_data WHERE result_file_name = ?", (result_file_name,)
        )
        result = cur.fetchone()

        if result is None:
            cur.execute(
                "INSERT INTO vehicle_data (location, vehicle_type, speed, license_plate, result_file_name, time) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    location,
                    vehicle_type,
                    speed,
                    license_plate,
                    result_file_name,
                    formatted_time,
                ),
            )

    conn.commit()

    cur.execute("SELECT * FROM vehicle_data")
    rows = cur.fetchall()

    data = []
    for row in rows:
        data.append(
            {
                "id": row[0],
                "location": row[1],
                "vehicle_type": row[2],
                "speed": row[3],
                "license_plate": row[4],
                "result_file_name": row[5],
                "time": row[6],
            }
        )

    conn.close()

    return {"data": data}, 200


@app.route("/extract_json", methods=["GET"])
def extract_json():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM vehicle_data")

    rows = [dict(row) for row in cur.fetchall()]

    with open("database.json", "w") as f:
        json.dump(rows, f)

    response = jsonify(message="Data successfully written to database.json")
    response.headers["Content-Disposition"] = "attachment; filename=database.json"
    response.headers["X-Accel-Redirect"] = "database.json"
    return response


@app.route("/api/get_image/<path:path>")
def send_report(path):
    return send_from_directory("static", path)


@app.route("/api/get_databasejson")
def serve_database():
    return send_from_directory("", "database.json")


@app.route("/api/search", methods=["POST", "GET"])
def search():
    data = request.get_json(force=True)

    location = data.get("location")
    licensePlate = data.get("licensePlate")
    speedRange = data.get("speedRange")
    date = data.get("date")
    timeRange = data.get("timeRange")

    if not speedRange or not timeRange or "" in speedRange or "" in timeRange:
        return {"data": "Time stamp must be filled out"}, 400

    # Connect to the SQLite database
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    print("location: ", location)
    print("licensePlate: ", licensePlate)
    print("speedRangeMIN: ", speedRange[0])
    print("speedRangeMAX: ", speedRange[1])

    date = "-".join(reversed(date.split("-")))

    print("date: ", date)

    # Convert date and timeRange into datetime strings
    startDateTime = datetime.strptime(date + " " + timeRange[0], "%d-%m-%Y %H:%M:%S")
    endDateTime = datetime.strptime(date + " " + timeRange[1], "%d-%m-%Y %H:%M:%S")

    # Convert datetime objects back to strings
    startDateTimeStr = startDateTime.strftime("%d-%m-%Y %H:%M:%S")
    endDateTimeStr = endDateTime.strftime("%d-%m-%Y %H:%M:%S")

    print("startDateTimeStr: ", startDateTimeStr)
    print("endDateTimeStr: ", endDateTimeStr)

    # Convert speedRange values to strings
    speedRangeStr = [str(speed) for speed in speedRange]

    # Create a query string with placeholders for the parameters
    query = """
        SELECT * FROM vehicle_data
        WHERE location LIKE ?
        AND license_plate LIKE ?
        AND (CAST(speed AS REAL) BETWEEN ? AND ? OR speed = 'Not Detected')
        AND time BETWEEN ? AND ?
    """

    # Execute the query with the parameters
    cur.execute(
        query,
        (
            "%" + location + "%",
            "%" + licensePlate + "%",
            speedRangeStr[0],
            speedRangeStr[1],
            startDateTimeStr,
            endDateTimeStr,
        ),
    )

    # Fetch all the rows
    rows = cur.fetchall()
    print("-------ROWS-------")

    # Print the rows
    print(rows)

    # Get the column names from the cursor description
    column_names = [column[0] for column in cur.description]

    # Convert the rows into a list of dictionaries
    rows = [dict(zip(column_names, row)) for row in rows]

    if not rows:
        return {"data": "Don't have record"}, 200

    return {"data": rows}, 200


if __name__ == "__main__":
    app.run(debug=True)
