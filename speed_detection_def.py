from datetime import datetime


def euclidean_distance(point1: tuple, point2: tuple):
    x1, y1 = point1
    x2, y2 = point2
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance


kmh_to_ms_const = 3.6
pixel_to_meter_const = 0.1


def calculate_avg_speed(
    track_id, cross_yellow_line, cross_orange_line, cross_pink_line
):
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
