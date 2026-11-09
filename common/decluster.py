"""Gruppiert Beben zu Bebenserien (gleiche Regel fuer Feature- und Inference-Pipeline)."""

import numpy as np

from common.config import DECLUSTER_SPACE_KM, DECLUSTER_TIME_DAYS, MAINSHOCK_MIN_MAGNITUDE


def haversine_km(lat1, lon1, lat2, lon2):
    """Distanz zwischen zwei Punkten auf der Erdkugel, in km."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def decluster_sequences(events, space_km=DECLUSTER_SPACE_KM, time_days=DECLUSTER_TIME_DAYS,
                         mainshock_min_magnitude=MAINSHOCK_MIN_MAGNITUDE):
    """Ordnet jedem Beben eine Serie zu (Spalte mainshock_id); Beben ohne Serie bleiben None."""
    events = events.copy()
    events["mainshock_id"] = None
    active_sequences = []

    for idx, row in events.iterrows():
        matched = None
        for seq in active_sequences:
            days_since = (row["zeit"] - seq["last_zeit"]).total_seconds() / 86400
            if days_since > time_days:
                continue
            if haversine_km(row["lat"], row["lon"], seq["lat"], seq["lon"]) <= space_km:
                matched = seq
                break

        if matched is None:
            if row["magnitude"] < mainshock_min_magnitude:
                continue
            new_id = row["event_id"]
            events.at[idx, "mainshock_id"] = new_id
            active_sequences.append({
                "mainshock_id": new_id, "mainshock_mag": row["magnitude"],
                "last_zeit": row["zeit"], "lat": row["lat"], "lon": row["lon"],
                "row_indices": [idx],
            })
        else:
            if row["magnitude"] >= matched["mainshock_mag"]:
                matched["mainshock_id"] = row["event_id"]
                matched["mainshock_mag"] = row["magnitude"]
                events.loc[matched["row_indices"], "mainshock_id"] = matched["mainshock_id"]
            matched["row_indices"].append(idx)
            events.at[idx, "mainshock_id"] = matched["mainshock_id"]
            matched["last_zeit"] = row["zeit"]
            matched["lat"], matched["lon"] = row["lat"], row["lon"]

    return events
