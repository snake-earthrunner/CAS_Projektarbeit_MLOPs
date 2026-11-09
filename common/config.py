"""Gemeinsame Einstellungen fuer alle Pipelines."""

USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
FETCH_MIN_MAGNITUDE = 3.0
MAINSHOCK_MIN_MAGNITUDE = 5.0
START_TIME = "2001-01-01"

MIN_LAT, MAX_LAT = 24.0, 46.0
MIN_LON, MAX_LON = 122.0, 146.0

DEMO_RESERVE_IDS = [
    "us2000h8ty",  # Hokkaido-Iburi 2018
    "us10007b88",  # Fukushima/Namie 2016
    "us20005iis",  # Kumamoto 2016
]

SNAPSHOT_INTERVAL_HOURS = 6
SNAPSHOT_WINDOW_DAYS = 14

LABEL_WINDOW_HOURS = 6
LABEL_MIN_MAGNITUDE = 4.5

EARLY_AFTERSHOCK_WINDOW_MINUTES = 60
EARLY_AFTERSHOCK_MIN_MAGNITUDE = 3.0

GR_A, GR_B = 4.8, 1.5

DECLUSTER_SPACE_KM = 100
DECLUSTER_TIME_DAYS = 14

HOPSWORKS_FEATURE_GROUP_NAME = "nachbeben_snapshots"
HOPSWORKS_FEATURE_GROUP_VERSION = 1

HOPSWORKS_FEATURE_VIEW_NAME = "nachbeben_snapshots_view"
HOPSWORKS_FEATURE_VIEW_VERSION = 1

FEATURE_COLUMNS = [
    "kumulative_energie",
    "b_wert",
    "zeit_seit_letztem_beben_h",
    "letzte_magnitude",
    "letzte_tiefe_km",
    "anzahl_frueh_nachbeben",
]
LABEL_COLUMN = "label_m45_naechste_6h"

TRAIN_START_DATE = "2013-01-01"

B_MC = 4.5
B_FENSTER_TAGE = 365
B_MIN_EREIGNISSE = 200
B_MIN, B_MAX = 0.6, 1.6

HOPSWORKS_B_FG_NAME = "regionaler_b_wert"
HOPSWORKS_B_FG_VERSION = 1
