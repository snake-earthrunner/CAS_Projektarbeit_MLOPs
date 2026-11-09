"""Training Pipeline: Feature View -> logistische Regression -> model.joblib (lokal)."""

import os
import sys

# damit 'import common' beim Start aus dem Repo-Ordner geht
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.config import FEATURE_COLUMNS, LABEL_COLUMN, TRAIN_START_DATE

MODEL_OUT_PATH = "training_pipeline/output/model.joblib"


def load_feature_data():
    """Trainingsdatensatz aus der Feature View laden."""
    from common.hopsworks_io import read_training_data_df
    df = read_training_data_df()
    df["snapshot_zeit"] = pd.to_datetime(df["snapshot_zeit"], utc=True, format="ISO8601")
    return df


def exclude_demo_reserve(df):
    """Entfernt Demo-Reserve-Zeilen aus den Trainingsdaten."""
    return df[df["ist_demo_reserve"] == False].reset_index(drop=True)


def select_training_period(df):
    """Behaelt ganze Serien ab TRAIN_START_DATE."""
    start = df.groupby("mainshock_id")["snapshot_zeit"].transform("min")
    return df[start >= pd.Timestamp(TRAIN_START_DATE, tz="UTC")].reset_index(drop=True)


def prepare_training_data(df):
    """Entfernt Zeilen mit fehlenden Features oder ohne Label."""
    return df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN]).reset_index(drop=True)


def split_sequences(df, test_size=0.2, random_state=42):
    """Teilt nach Serien (nicht nach Zeilen) in Training und Pruefung."""
    sequence_info = (
        df.groupby("mainshock_id")[LABEL_COLUMN]
        .max()
        .reset_index()
        .rename(columns={LABEL_COLUMN: "sequenz_hat_positiv"})
    )
    train_ids, val_ids = train_test_split(
        sequence_info["mainshock_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=sequence_info["sequenz_hat_positiv"],
    )
    return set(train_ids), set(val_ids)


def main():
    print("1/5 Feature-Daten laden...")
    df = load_feature_data()
    print(f"    {len(df)} Zeilen geladen")

    print("2/5 Demo-Reserve ausschliessen...")
    df = exclude_demo_reserve(df)
    print(f"    {len(df)} Zeilen übrig")
    df = select_training_period(df)
    print(f"    {len(df)} Zeilen ({df['mainshock_id'].nunique()} Sequenzen) ab {TRAIN_START_DATE}")

    df = prepare_training_data(df)
    print(f"    {len(df)} Zeilen nach Entfernen unvollständiger Snapshots")

    X = df[FEATURE_COLUMNS]
    y = df[LABEL_COLUMN]
    print(f"    Klassenverteilung Label: {y.value_counts().to_dict()}")

    print("3/5 Train/Val-Split (nach Sequenz gruppiert, stratifiziert)...")
    train_ids, val_ids = split_sequences(df)

    os.makedirs("training_pipeline/output", exist_ok=True)
    pd.DataFrame({
        "mainshock_id": list(train_ids) + list(val_ids),
        "split": ["train"] * len(train_ids) + ["val"] * len(val_ids),
    }).to_csv("training_pipeline/output/train_val_split.csv", index=False)

    train_mask = df["mainshock_id"].isin(train_ids)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]
    print(f"    Train: {len(X_train)} Zeilen ({len(train_ids)} Sequenzen), "
          f"Val: {len(X_val)} Zeilen ({len(val_ids)} Sequenzen)")

    print("4/5 Modell trainieren + evaluieren...")
    # Scaler nur auf den Trainingsdaten lernen (sonst Data Leak)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    majority_class = y_train.mode()[0]
    baseline_pred = pd.Series(majority_class, index=y_val.index)
    baseline_acc = accuracy_score(y_val, baseline_pred)
    baseline_f1 = f1_score(y_val, baseline_pred, zero_division=0)
    print(f"    Baseline (immer '{majority_class}' vorhersagen): Accuracy: {baseline_acc:.3f}, F1: {baseline_f1:.3f}")
    print(f"    Modell:                                          Accuracy: {acc:.3f}, F1: {f1:.3f}")

    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    print(f"    ROC-AUC:                                         {auc:.3f} (0.5 = raet nur, 1.0 = perfekt)")

    print("5/5 Modell speichern...")
    os.makedirs("training_pipeline/output", exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_columns": FEATURE_COLUMNS}, MODEL_OUT_PATH)
    print(f"\nFertig. Modell gespeichert nach {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
