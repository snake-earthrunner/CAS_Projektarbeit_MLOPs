"""Feature- und Label-Berechnung, gemeinsam fuer Feature- und Inference-Pipeline."""

from datetime import timedelta

import numpy as np
import pandas as pd

from common.config import (
    GR_A, GR_B,
    EARLY_AFTERSHOCK_WINDOW_MINUTES, EARLY_AFTERSHOCK_MIN_MAGNITUDE,
    LABEL_WINDOW_HOURS, LABEL_MIN_MAGNITUDE,
    B_MC, B_FENSTER_TAGE, B_MIN_EREIGNISSE, B_MIN, B_MAX,
)


def compute_aggregated_feature(sequence_events, snapshot_zeit):
    """Kumulative Energie aller Beben der Serie bis snapshot_zeit."""
    known = sequence_events[sequence_events["zeit"] <= snapshot_zeit]
    if known.empty:
        return 0.0
    return float(np.sum(10 ** (GR_B * known["magnitude"] + GR_A)))


def compute_rt_features(sequence_events, snapshot_zeit):
    """Die vier RT-Features aus den Beben bis snapshot_zeit."""
    known = sequence_events[sequence_events["zeit"] <= snapshot_zeit]

    if known.empty:
        return {
            "zeit_seit_letztem_beben_h": np.nan,
            "letzte_magnitude": np.nan,
            "letzte_tiefe_km": np.nan,
            "anzahl_frueh_nachbeben": 0,
        }

    letztes = known.iloc[-1]
    zeit_seit_letztem_h = (snapshot_zeit - letztes["zeit"]).total_seconds() / 3600

    mainshock_id_value = sequence_events["mainshock_id"].iloc[0]
    mainshock_row = sequence_events[sequence_events["event_id"] == mainshock_id_value]
    if mainshock_row.empty:
        mainshock_zeit = known["zeit"].min()
    else:
        mainshock_zeit = mainshock_row["zeit"].iloc[0]
    fenster_ende = mainshock_zeit + timedelta(minutes=EARLY_AFTERSHOCK_WINDOW_MINUTES)
    frueh = known[
        (known["zeit"] > mainshock_zeit) & (known["zeit"] <= fenster_ende)
        & (known["magnitude"] >= EARLY_AFTERSHOCK_MIN_MAGNITUDE)
    ]

    return {
        "zeit_seit_letztem_beben_h": zeit_seit_letztem_h,
        "letzte_magnitude": float(letztes["magnitude"]),
        "letzte_tiefe_km": float(letztes["tiefe_km"]),
        "anzahl_frueh_nachbeben": int(len(frueh)),
    }


def compute_label(sequence_events, snapshot_zeit, window_hours=LABEL_WINDOW_HOURS,
                   min_magnitude=LABEL_MIN_MAGNITUDE):
    """1, wenn in den naechsten window_hours ein Beben ab min_magnitude folgt, sonst 0 (nie im Live-Modus)."""
    fenster_ende = snapshot_zeit + timedelta(hours=window_hours)
    future = sequence_events[
        (sequence_events["zeit"] > snapshot_zeit)
        & (sequence_events["zeit"] <= fenster_ende)
        & (sequence_events["magnitude"] >= min_magnitude)
    ]
    return int(len(future) > 0)


def compute_regionaler_b_wert(katalog, tage, ab_datum=None):
    """b-Wert pro Tag D aus den Beben der 365 Tage vor D (ohne Tag D)."""
    k = katalog[katalog["magnitude"] >= B_MC].sort_values("zeit")
    zeiten = k["zeit"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    mags = k["magnitude"].round(1).to_numpy()

    ab = None
    if ab_datum is not None:
        ab = pd.Timestamp(ab_datum)
        ab = ab.tz_convert("UTC") if ab.tzinfo else ab.tz_localize("UTC")
        ab = np.datetime64(ab.tz_localize(None))

    zeilen = []
    for tag in tage:
        tag = pd.Timestamp(tag)
        tag = tag.tz_convert("UTC") if tag.tzinfo else tag.tz_localize("UTC")
        bis = np.datetime64(tag.tz_localize(None))
        von = np.datetime64((tag - pd.Timedelta(days=B_FENSTER_TAGE)).tz_localize(None))
        i0 = np.searchsorted(zeiten, von, side="left")
        i1 = np.searchsorted(zeiten, bis, side="left")
        m = mags[i0:i1]
        n = len(m)
        b = np.nan
        fenster_komplett = ab is None or von >= ab
        if fenster_komplett and n >= B_MIN_EREIGNISSE:
            # Aki/Utsu-Schaetzer
            b = np.log10(np.e) / (m.mean() - (B_MC - 0.05))
            if not (B_MIN <= b <= B_MAX):
                b = np.nan
        zeilen.append({"tag": tag, "b_wert": b, "n_ereignisse": n})
    return pd.DataFrame(zeilen)
