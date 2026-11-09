"""Kontrolle der Feature View: keine Duplikate, richtiger Tag, b-Wert von Hand nachgerechnet."""

import math
import os
import sys

# damit 'import common' beim Start aus dem Repo-Ordner geht
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from common.config import (
    START_TIME, FETCH_MIN_MAGNITUDE, B_MC, B_FENSTER_TAGE, TRAIN_START_DATE,
)
from common.usgs import fetch_usgs_events
from common.hopsworks_io import read_feature_view_df


def b_von_hand(katalog, tag):
    """b-Wert unabhaengig von common.features nachrechnen."""
    von = tag - pd.Timedelta(days=B_FENSTER_TAGE)
    m = [round(r.magnitude, 1) for r in katalog.itertuples()
         if r.magnitude >= B_MC and von <= r.zeit < tag]
    return math.log10(math.e) / (sum(m) / len(m) - (B_MC - 0.05)), len(m)


def main():
    df = read_feature_view_df()
    df["snapshot_zeit"] = pd.to_datetime(df["snapshot_zeit"], utc=True)
    df["tag"] = pd.to_datetime(df["tag"], utc=True)
    start = pd.Timestamp(TRAIN_START_DATE, tz="UTC")
    print(f"Zeilen in der Feature View: {len(df)}")

    dup = int(df.duplicated(subset=["mainshock_id", "snapshot_zeit"]).sum())
    print(f"1) Doppelte Zeilen durch den Join: {dup} (erwartet: 0)")

    falsch = int((df["tag"] != df["snapshot_zeit"].dt.floor("D")).sum())
    print(f"2) Zeilen mit falschem tag: {falsch} (erwartet: 0)")

    ab = df[df["snapshot_zeit"] >= start]
    vor = df[df["snapshot_zeit"] < start]
    print(f"   ab {TRAIN_START_DATE}: {len(ab)} Zeilen, ohne b_wert: {int(ab['b_wert'].isna().sum())} (erwartet: 0)")
    print(f"   vor {TRAIN_START_DATE}: {len(vor)} Zeilen, mit b_wert: {int(vor['b_wert'].notna().sum())} (erwartet: 0)")

    print("3) Drei zufaellige Zeilen von Hand nachgerechnet (Katalog wird geladen)...")
    katalog = fetch_usgs_events(start=START_TIME, min_magnitude=FETCH_MIN_MAGNITUDE)
    for r in ab.sample(3, random_state=1).itertuples():
        b, n = b_von_hand(katalog, r.snapshot_zeit.floor("D"))
        ok = abs(b - r.b_wert) < 1e-4
        print(f"   {r.snapshot_zeit}  Store: {r.b_wert:.4f}  von Hand: {b:.4f} (N={n})  -> {'OK' if ok else 'FEHLER'}")


if __name__ == "__main__":
    main()
