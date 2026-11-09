"""Feature Pipeline: USGS -> Bebenserien -> Features + Label -> Hopsworks."""

import os
import sys

# damit 'import common' beim Start aus dem Repo-Ordner geht
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.config import (
    START_TIME, FETCH_MIN_MAGNITUDE, DEMO_RESERVE_IDS,
    SNAPSHOT_INTERVAL_HOURS, SNAPSHOT_WINDOW_DAYS, LABEL_WINDOW_HOURS, LABEL_COLUMN,
    TRAIN_START_DATE,
)
from common.usgs import fetch_usgs_events
from common.decluster import decluster_sequences
from common.features import (
    compute_aggregated_feature, compute_rt_features, compute_label,
    compute_regionaler_b_wert,
)
from common.hopsworks_io import (
    write_feature_group, write_b_wert_feature_group, create_joined_feature_view,
)

import pandas as pd
from datetime import timedelta


def identify_demo_reserve_sequences(events, reserve_ids=None):
    """Gibt die mainshock_ids der Demo-Beben zurueck."""
    if reserve_ids is None:
        reserve_ids = DEMO_RESERVE_IDS
    reserved_rows = events[events["event_id"].isin(reserve_ids)]
    return set(reserved_rows["mainshock_id"].unique())


def build_snapshots(events, interval_hours=SNAPSHOT_INTERVAL_HOURS, window_days=SNAPSHOT_WINDOW_DAYS,
                     label_window_hours=LABEL_WINDOW_HOURS, now=None):
    """Messpunkte pro Serie: ab dem Hauptbeben alle 6 h, 14 Tage lang, nur bis jetzt."""
    if now is None:
        now = pd.Timestamp.now(tz="UTC")

    rows = []
    n_steps = int((window_days * 24) / interval_hours)
    for mainshock_id, group in events.groupby("mainshock_id"):
        mainshock_row = group[group["event_id"] == mainshock_id]
        if mainshock_row.empty:
            print(f"    WARNUNG: Hauptbeben-Ereignis fuer Sequenz {mainshock_id} nicht gefunden, uebersprungen.")
            continue
        start = mainshock_row["zeit"].iloc[0]
        for step in range(n_steps + 1):
            snapshot_zeit = start + timedelta(hours=step * interval_hours)
            if snapshot_zeit > now:
                break
            label_bekannt = (snapshot_zeit + timedelta(hours=label_window_hours)) <= now
            rows.append({
                "mainshock_id": mainshock_id,
                "snapshot_zeit": snapshot_zeit,
                "label_bekannt": label_bekannt,
            })
    return pd.DataFrame(rows)


def main():
    print("1/7 Rohdaten von USGS holen...")
    events = fetch_usgs_events(start=START_TIME, min_magnitude=FETCH_MIN_MAGNITUDE)
    print(f"    {len(events)} Beben gefunden (M>={FETCH_MIN_MAGNITUDE}, Japan, {START_TIME} bis heute)")

    katalog = events

    print("2/7 Sequenzen deklustern (alle Beben, inkl. Demo-Reserve)...")
    events = decluster_sequences(events)
    n_sequences = events["mainshock_id"].nunique()
    print(f"    {n_sequences} Sequenzen erkannt")

    print("3/7 Demo-Reserve-Sequenzen markieren...")
    reserve_mainshock_ids = identify_demo_reserve_sequences(events)
    print(f"    {len(reserve_mainshock_ids)} Demo-Reserve-Sequenz(en) gefunden: {reserve_mainshock_ids}")
    if len(reserve_mainshock_ids) == 0:
        print("    WARNUNG: keine Demo-Beben gefunden.")

    print("4/7 Snapshots bauen...")
    snapshots = build_snapshots(events)
    print(f"    {len(snapshots)} Snapshot-Zeitpunkte")

    print("5/7 Features + Label berechnen...")
    feature_rows = []
    for mainshock_id, seq_snapshots in snapshots.groupby("mainshock_id"):
        sequence_events = events[events["mainshock_id"] == mainshock_id]
        ist_demo_reserve = mainshock_id in reserve_mainshock_ids
        for _, snap in seq_snapshots.iterrows():
            snapshot_zeit = snap["snapshot_zeit"]
            energie = compute_aggregated_feature(sequence_events, snapshot_zeit)
            rt = compute_rt_features(sequence_events, snapshot_zeit)
            # Label nur, wenn das 6-h-Fenster schon vorbei ist
            if snap["label_bekannt"]:
                label = compute_label(sequence_events, snapshot_zeit)
            else:
                label = None
            feature_rows.append({
                "mainshock_id": mainshock_id,
                "snapshot_zeit": snapshot_zeit,
                "kumulative_energie": energie,
                **rt,
                LABEL_COLUMN: label,
                "ist_demo_reserve": ist_demo_reserve,
            })
    result = pd.DataFrame(feature_rows)
    result["tag"] = result["snapshot_zeit"].dt.floor("D")

    print("6/7 Als CSV speichern...")
    os.makedirs("feature_pipeline/output", exist_ok=True)
    out_path = "feature_pipeline/output/snapshots.csv"
    result.to_csv(out_path, index=False)

    expected = n_sequences * (int((SNAPSHOT_WINDOW_DAYS * 24) / SNAPSHOT_INTERVAL_HOURS) + 1)
    n_reserve_rows = int(result["ist_demo_reserve"].sum())
    print(f"\nFertig. {len(result)} Zeilen geschrieben nach {out_path}")
    print(f"Zur Kontrolle: {n_sequences} Sequenzen -> ungefähr {expected} Zeilen erwartet")
    print(f"Davon Demo-Reserve (nie fürs Training): {n_reserve_rows} Zeilen")

    print("7/7 Regionalen b-Wert berechnen (ein Wert pro Tag ab Trainingsbeginn)...")
    tage = pd.date_range(
        pd.Timestamp(TRAIN_START_DATE, tz="UTC"),
        pd.Timestamp.now(tz="UTC").normalize(),
        freq="D",
    )
    b_df = compute_regionaler_b_wert(katalog, tage, ab_datum=START_TIME)
    print(f"    b-Wert: {len(b_df)} Tage, davon {int(b_df['b_wert'].isna().sum())} ohne Wert (NaN)")

    print("    In Hopsworks schreiben: b-Wert-Tabelle, Snapshot-Tabelle, Feature View...")
    write_b_wert_feature_group(b_df)
    write_feature_group(result)
    create_joined_feature_view()
    print("    Fertig -- beide Feature Groups und die Feature View in Hopsworks aktualisiert.")


if __name__ == "__main__":
    main()
