"""Abruf der Beben bei der USGS-API."""

from datetime import datetime, timezone

import pandas as pd
import requests

from common.config import USGS_URL, FETCH_MIN_MAGNITUDE, MIN_LAT, MAX_LAT, MIN_LON, MAX_LON


def fetch_usgs_events(start, end=None, min_magnitude=FETCH_MIN_MAGNITUDE,
                       min_lat=MIN_LAT, max_lat=MAX_LAT, min_lon=MIN_LON, max_lon=MAX_LON):
    """Holt Beben von USGS als DataFrame (event_id, zeit, magnitude, tiefe_km, lat, lon)."""
    if end is None:
        # mit Uhrzeit, sonst fehlen Beben von heute
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    rows = _fetch_chunk(start, end, min_magnitude, min_lat, max_lat, min_lon, max_lon)
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset="event_id")
        .sort_values("zeit")
        .reset_index(drop=True)
    )


def _fetch_chunk(start, end, min_magnitude, min_lat, max_lat, min_lon, max_lon):
    """Holt einen Zeitabschnitt; bei mehr als 20'000 Treffern wird er halbiert."""
    params = {
        "format": "geojson",
        "starttime": start,
        "endtime": end,
        "minmagnitude": min_magnitude,
        "minlatitude": min_lat,
        "maxlatitude": max_lat,
        "minlongitude": min_lon,
        "maxlongitude": max_lon,
        "orderby": "time-asc",
    }
    response = requests.get(USGS_URL, params=params, timeout=60)

    if response.status_code == 400 and "exceeds search limit" in response.text:
        start_dt, end_dt = pd.to_datetime(start), pd.to_datetime(end)
        mid = (start_dt + (end_dt - start_dt) / 2).strftime("%Y-%m-%d")
        return (
            _fetch_chunk(start, mid, min_magnitude, min_lat, max_lat, min_lon, max_lon)
            + _fetch_chunk(mid, end, min_magnitude, min_lat, max_lat, min_lon, max_lon)
        )

    response.raise_for_status()
    data = response.json()

    rows = []
    for feature in data["features"]:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        rows.append({
            "event_id": feature["id"],
            "zeit": pd.to_datetime(props["time"], unit="ms", utc=True),
            "magnitude": props["mag"],
            "tiefe_km": coords[2],
            "lat": coords[1],
            "lon": coords[0],
        })
    return rows
