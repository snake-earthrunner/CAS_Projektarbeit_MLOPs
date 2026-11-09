"""Inference Pipeline: Modus live oder replay, gibt die Wahrscheinlichkeit aus."""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

# damit 'import common' beim Start aus dem Repo-Ordner geht
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd

from common.config import (
    START_TIME, FETCH_MIN_MAGNITUDE, MAINSHOCK_MIN_MAGNITUDE, SNAPSHOT_WINDOW_DAYS, FEATURE_COLUMNS,
)
from common.usgs import fetch_usgs_events
from common.decluster import decluster_sequences
from common.features import compute_rt_features, compute_label, compute_aggregated_feature
from common.config import LABEL_WINDOW_HOURS, LABEL_MIN_MAGNITUDE

MODEL_PATH = "training_pipeline/output/model.joblib"


def get_live_target_sequence():
    """Neueste Serie der letzten 21 Tage, Zeitpunkt = jetzt."""
    recent_start = (datetime.now(timezone.utc) - timedelta(days=SNAPSHOT_WINDOW_DAYS + 7)).strftime("%Y-%m-%d")
    events = fetch_usgs_events(start=recent_start, min_magnitude=FETCH_MIN_MAGNITUDE)
    if events.empty:
        raise RuntimeError("Keine aktuellen Beben gefunden -- 'live'-Modus geht gerade nicht, 'replay' verwenden.")

    events = decluster_sequences(events)
    mainshock_events = events[events["magnitude"] >= MAINSHOCK_MIN_MAGNITUDE]
    if mainshock_events.empty:
        raise RuntimeError(
            "Kein Hauptbeben (M>=5.0) in den letzten Tagen gefunden -- "
            "'live'-Modus geht gerade nicht, 'replay' verwenden."
        )

    latest_event = mainshock_events.iloc[-1]
    mainshock_id = latest_event["mainshock_id"]
    sequence_events = events[events["mainshock_id"] == mainshock_id]
    snapshot_zeit = pd.Timestamp(datetime.now(timezone.utc))
    return mainshock_id, sequence_events, snapshot_zeit


def get_replay_target_sequence(mainshock_id, hours_after_mainshock=0):
    """Serie eines vergangenen Hauptbebens, Zeitpunkt = Hauptbeben + hours_after_mainshock."""
    events = fetch_usgs_events(start=START_TIME, min_magnitude=FETCH_MIN_MAGNITUDE)
    events = decluster_sequences(events)
    sequence_events = events[events["mainshock_id"] == mainshock_id]
    if sequence_events.empty:
        raise ValueError(f"mainshock_id '{mainshock_id}' nach dem Deklustern nicht gefunden.")

    mainshock_row = sequence_events[sequence_events["event_id"] == mainshock_id]
    if mainshock_row.empty:
        raise ValueError(f"Hauptbeben '{mainshock_id}' nicht in den Daten gefunden.")
    mainshock_zeit = mainshock_row["zeit"].iloc[0]
    snapshot_zeit = mainshock_zeit + timedelta(hours=hours_after_mainshock)
    return mainshock_id, sequence_events, snapshot_zeit


def read_aggregated_features_from_store(mainshock_id, snapshot_zeit):
    """Liest Energie und b-Wert der neuesten Zeile dieser Serie aus der Feature View."""
    from common.hopsworks_io import read_feature_view_df
    df = read_feature_view_df()
    df["snapshot_zeit"] = pd.to_datetime(df["snapshot_zeit"], utc=True)
    subset = df[(df["mainshock_id"] == mainshock_id) & (df["snapshot_zeit"] <= snapshot_zeit)]
    if subset.empty:
        raise RuntimeError(
            f"Kein Feature-Store-Eintrag für mainshock_id '{mainshock_id}' bis {snapshot_zeit} gefunden. "
            "Das aggregierte Feature ist noch nicht berechnet -- Feature Pipeline zuerst (erneut) "
            "laufen lassen: python feature_pipeline/main.py"
        )
    neuester = subset.sort_values("snapshot_zeit").iloc[-1]
    if pd.isna(neuester["b_wert"]):
        raise RuntimeError(
            f"Kein b-Wert im Store für {neuester['snapshot_zeit']} (Tag ohne Wert). "
            "Feature Pipeline zuerst (erneut) laufen lassen: python feature_pipeline/main.py"
        )
    return float(neuester["kumulative_energie"]), float(neuester["b_wert"])


def build_feature_vector(aggregated_energie, b_wert, rt_features):
    """Feature-Vektor in der Reihenfolge FEATURE_COLUMNS."""
    values = {"kumulative_energie": aggregated_energie, "b_wert": b_wert, **rt_features}
    return pd.DataFrame([{col: values[col] for col in FEATURE_COLUMNS}])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "replay"], default="live")
    parser.add_argument("--mainshock-id", default=None, help="Nur für --mode replay")
    parser.add_argument("--6h-fenster-start-nach-hauptbeben", dest="stunden_nach_hauptbeben",
                         type=float, default=0,
                         help="Nur für --mode replay: WANN startet das feste 6h-Vorhersage-Fenster, "
                              "in Stunden nach dem Hauptbeben (Standard: 0 = Fenster startet direkt "
                              "beim Hauptbeben, laeuft bis Stunde 6). Beispiel: der Wert 6 bedeutet "
                              "'6h-Fenster von Stunde 6 bis Stunde 12 nach dem Hauptbeben'.")
    args = parser.parse_args()

    print(f"1/5 Ziel-Sequenz bestimmen (Modus: {args.mode})...")
    if args.mode == "live":
        mainshock_id, sequence_events, snapshot_zeit = get_live_target_sequence()
    else:
        if not args.mainshock_id:
            raise SystemExit("--mode replay braucht --mainshock-id <event_id> (aus der Demo-Reserve)")
        mainshock_id, sequence_events, snapshot_zeit = get_replay_target_sequence(
            args.mainshock_id, args.stunden_nach_hauptbeben
        )
    print(f"    mainshock_id: {mainshock_id}, snapshot_zeit: {snapshot_zeit}, "
          f"{len(sequence_events)} bekannte Beben in der Sequenz")
    if args.mode == "replay":
        start_h = args.stunden_nach_hauptbeben
        ende_h = start_h + LABEL_WINDOW_HOURS
        print(f"    -> Frage: Passiert ein M4.5+-Beben zwischen Stunde {start_h:.0f} und "
              f"Stunde {ende_h:.0f} nach dem Hauptbeben?")
    else:
        print(f"    -> Frage: Passiert ein M4.5+-Beben in den naechsten "
              f"{LABEL_WINDOW_HOURS:.0f}h ab jetzt?")

    if args.mode == "live":
        print("2/5 Aggregierte Features (Energie, b-Wert) aus dem Store lesen...")
        aggregated_energie, b_wert = read_aggregated_features_from_store(mainshock_id, snapshot_zeit)
        print(f"    kumulative_energie = {aggregated_energie:.3e}, b_wert = {b_wert:.3f}")
    else:
        print("2/5 Aggregiertes Feature exakt fuer den Replay-Zeitpunkt berechnen...")
        # Replay: Energie live nachrechnen, b-Wert aus dem Store
        aggregated_energie = compute_aggregated_feature(sequence_events, snapshot_zeit)
        _, b_wert = read_aggregated_features_from_store(mainshock_id, snapshot_zeit)
        print(f"    kumulative_energie = {aggregated_energie:.3e}, b_wert = {b_wert:.3f}")

    print("3/5 RT-Features live berechnen...")
    rt_features = compute_rt_features(sequence_events, snapshot_zeit)
    print(f"    {rt_features}")

    print("4/5 Feature-Vektor zusammenbauen...")
    X = build_feature_vector(aggregated_energie, b_wert, rt_features)
    print(f"    {X.to_dict('records')[0]}")

    print("5/5 Modell laden + Vorhersage...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} nicht gefunden -- zuerst: python training_pipeline/main.py")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    X_scaled = scaler.transform(X)
    probability = model.predict_proba(X_scaled)[0][1]

    if args.mode == "replay":
        start_h = args.stunden_nach_hauptbeben
        ende_h = start_h + LABEL_WINDOW_HOURS
        print(f"\nWahrscheinlichkeit fuer M4.5+ zwischen Stunde {start_h:.0f} und "
              f"Stunde {ende_h:.0f} nach dem Hauptbeben: {probability:.2%}")
    else:
        print(f"\nWahrscheinlichkeit fuer M4.5+ in den naechsten {LABEL_WINDOW_HOURS:.0f}h ab jetzt: {probability:.2%}")

    if args.mode == "replay":
        tatsaechlich = compute_label(sequence_events, snapshot_zeit)
        fenster_ende = snapshot_zeit + timedelta(hours=LABEL_WINDOW_HOURS)
        treffer = sequence_events[
            (sequence_events["zeit"] > snapshot_zeit)
            & (sequence_events["zeit"] <= fenster_ende)
            & (sequence_events["magnitude"] >= LABEL_MIN_MAGNITUDE)
        ].sort_values("zeit")
        print("Echte Daten (Ground Truth, nur bei Replay moeglich):")
        if tatsaechlich:
            erstes = treffer.iloc[0]
            stunden_bis = (erstes["zeit"] - snapshot_zeit).total_seconds() / 3600
            print(f"    Tatsächlich passiert: JA -- M{erstes['magnitude']:.1f} nach {stunden_bis:.1f}h")
        else:
            print("    Tatsächlich passiert: NEIN")


if __name__ == "__main__":
    main()
