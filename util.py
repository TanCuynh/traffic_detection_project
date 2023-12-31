import string
import easyocr

reader = easyocr.Reader(["en"], gpu=True)

dict_char_to_int = {
    "A": "4",
    "B": "8",
    "D": "0",
    "E": "6",
    "G": "6",
    "H": "4",
    "I": "1",
    "J": "3",
    "L": "4",
    "O": "0",
    "Q": "0",
    "S": "5",
    "T": "4",
    "Z": "2",
}

dict_int_to_char = {
    "0": "O",
    "1": "I",
    "3": "J",
    "4": "A",
    "6": "G",
    "5": "S",
    "8": "B",
    "9": "B",
}


def filter_characters(text):
    allowed_characters = string.ascii_uppercase + string.digits

    filtered_text = "".join(char for char in text if char in allowed_characters)

    return filtered_text


def license_complies_format(text):
    if len(text) == 8:
        if (
            (
                text[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[0] in dict_char_to_int.keys()
            )
            and (
                text[1] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[1] in dict_char_to_int.keys()
            )
            and (
                text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()
            )
            and (
                text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[4] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[4] in dict_char_to_int.keys()
            )
            and (
                text[5] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[5] in dict_char_to_int.keys()
            )
            and (
                text[6] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[6] in dict_char_to_int.keys()
            )
            and (
                text[7] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[7] in dict_char_to_int.keys()
            )
        ):
            return True
        else:
            return False
    elif len(text) == 9:
        if (
            (
                text[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[0] in dict_char_to_int.keys()
            )
            and (
                text[1] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[1] in dict_char_to_int.keys()
            )
            and (
                text[2] in string.ascii_uppercase or text[2] in dict_int_to_char.keys()
            )
            and (
                text[3] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[3] in dict_char_to_int.keys()
            )
            and (
                text[4] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[4] in dict_char_to_int.keys()
            )
            and (
                text[5] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[5] in dict_char_to_int.keys()
            )
            and (
                text[6] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[6] in dict_char_to_int.keys()
            )
            and (
                text[7] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[7] in dict_char_to_int.keys()
            )
            and (
                text[8] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
                or text[8] in dict_char_to_int.keys()
            )
        ):
            return True
        else:
            return False
    else:
        return False


def format_license(text):
    if len(text) == 8:
        license_plate_ = ""
        mapping = {
            0: dict_char_to_int,
            1: dict_char_to_int,
            2: dict_int_to_char,
            3: dict_char_to_int,
            4: dict_char_to_int,
            5: dict_char_to_int,
            6: dict_char_to_int,
            7: dict_char_to_int,
        }
        for j in [0, 1, 2, 3, 4, 5, 6, 7]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        license_plate_ = license_plate_.replace("K", "X")

        return license_plate_
    else:
        license_plate_ = ""
        mapping = {
            0: dict_char_to_int,
            1: dict_char_to_int,
            2: dict_int_to_char,
            3: dict_char_to_int,
            4: dict_char_to_int,
            5: dict_char_to_int,
            6: dict_char_to_int,
            7: dict_char_to_int,
            8: dict_char_to_int,
        }
        for j in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        license_plate_ = license_plate_.replace("K", "X")

        return license_plate_


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    full_license_plate = ""
    text_license_plate = ""

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(" ", "")
        text_license_plate += text

    full_license_plate = filter_characters(text_license_plate)
    if license_complies_format(full_license_plate):
        return format_license(full_license_plate), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1


def prioritize_outermost_largest(detections):
    result = []

    detections.sort(key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    while detections:
        current_detection = detections.pop(0)
        x1, y1, x2, y2, _, _ = current_detection

        is_outermost = all(
            x1 >= other_x2 or x2 <= other_x1 or y1 >= other_y2 or y2 <= other_y1
            for (other_x1, other_y1, other_x2, other_y2, _, _) in detections
        )

        if is_outermost:
            result.append(current_detection)

    return result
