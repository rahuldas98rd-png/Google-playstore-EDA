CREATE TABLE apps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_name TEXT,
    category TEXT,
    rating REAL,
    reviews INTEGER,
    size_mb REAL,
    installs INTEGER,
    type TEXT,
    price REAL,
    content_rating TEXT,
    genres TEXT,
    last_updated DATE,
    current_version TEXT,
    android_version TEXT
);
