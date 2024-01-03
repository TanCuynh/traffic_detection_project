DROP TABLE IF EXISTS vehicle_data;

CREATE TABLE vehicle_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location TEXT NOT NULL,  
    vehicle_type TEXT NOT NULL,
    speed TEXT NOT NULL,
    license_plate TEXT NOT NULL,
    result_file_name TEXT NOT NULL,
    time TEXT NOT NULL
);