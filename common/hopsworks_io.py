"""Lesen und Schreiben im Hopsworks Feature Store (API-Key aus .env)."""

import os
from dotenv import load_dotenv
import hopsworks

from common.config import (
    HOPSWORKS_FEATURE_GROUP_NAME, HOPSWORKS_FEATURE_GROUP_VERSION,
    HOPSWORKS_FEATURE_VIEW_NAME, HOPSWORKS_FEATURE_VIEW_VERSION,
    HOPSWORKS_B_FG_NAME, HOPSWORKS_B_FG_VERSION,
)


def get_feature_store():
    """Login bei Hopsworks, gibt den Feature Store zurueck."""
    load_dotenv()
    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT_NAME"],
    )
    return project.get_feature_store()


def write_feature_group(df):
    """Schreibt die Messpunkte neu; loescht vorher Feature View und Feature Group, sonst bleiben alte Zeilen."""
    fs = get_feature_store()
    try:
        old_fv = fs.get_feature_view(
            name=HOPSWORKS_FEATURE_VIEW_NAME,
            version=HOPSWORKS_FEATURE_VIEW_VERSION,
        )
        old_fv.delete()
    except Exception:
        pass

    try:
        old_fg = fs.get_feature_group(
            name=HOPSWORKS_FEATURE_GROUP_NAME,
            version=HOPSWORKS_FEATURE_GROUP_VERSION,
        )
        old_fg.delete()
    except Exception:
        pass

    fg = fs.get_or_create_feature_group(
        name=HOPSWORKS_FEATURE_GROUP_NAME,
        version=HOPSWORKS_FEATURE_GROUP_VERSION,
        primary_key=["mainshock_id", "snapshot_zeit"],
        event_time="snapshot_zeit",
        time_travel_format="HUDI",
        description="Nachbeben-Snapshots: aggregiertes Feature, RT-Features, Label",
    )
    fg.insert(df)

    return fg


def write_b_wert_feature_group(df):
    """Schreibt den b-Wert pro Tag neu in die eigene Feature Group."""
    fs = get_feature_store()
    try:
        old_fv = fs.get_feature_view(
            name=HOPSWORKS_FEATURE_VIEW_NAME,
            version=HOPSWORKS_FEATURE_VIEW_VERSION,
        )
        if old_fv is not None:
            old_fv.delete()
    except Exception:
        pass

    try:
        old_fg = fs.get_feature_group(
            name=HOPSWORKS_B_FG_NAME,
            version=HOPSWORKS_B_FG_VERSION,
        )
        if old_fg is not None:
            old_fg.delete()
    except Exception:
        pass

    fg = fs.get_or_create_feature_group(
        name=HOPSWORKS_B_FG_NAME,
        version=HOPSWORKS_B_FG_VERSION,
        primary_key=["tag"],
        event_time="tag",
        time_travel_format="HUDI",
        description="Regionaler b-Wert Japan (365 Tage, Mc 4.5), ein Wert pro Tag",
    )
    fg.insert(df)
    return fg


def create_joined_feature_view():
    """Legt die Feature View an: Snapshots + b-Wert, verbunden ueber 'tag'."""
    fs = get_feature_store()
    fg = fs.get_feature_group(
        name=HOPSWORKS_FEATURE_GROUP_NAME,
        version=HOPSWORKS_FEATURE_GROUP_VERSION,
    )
    b_fg = fs.get_feature_group(
        name=HOPSWORKS_B_FG_NAME,
        version=HOPSWORKS_B_FG_VERSION,
    )
    query = fg.select_all().join(b_fg.select(["b_wert", "n_ereignisse"]), on=["tag"])
    return fs.create_feature_view(
        name=HOPSWORKS_FEATURE_VIEW_NAME,
        version=HOPSWORKS_FEATURE_VIEW_VERSION,
        query=query,
    )


def get_or_create_feature_view():
    """Holt die Feature View oder legt sie an."""
    fs = get_feature_store()
    fv = None
    try:
        fv = fs.get_feature_view(
            name=HOPSWORKS_FEATURE_VIEW_NAME,
            version=HOPSWORKS_FEATURE_VIEW_VERSION,
        )
    except Exception:
        pass

    if fv is not None:
        return fv

    return create_joined_feature_view()


def read_feature_view_df():
    """Liest alle Zeilen der Feature View (fuer die Inferenz)."""
    fv = get_or_create_feature_view()
    return fv.get_batch_data()


def read_training_data_df():
    """Erstellt den Trainingsdatensatz aus der Feature View mit training_data()."""
    fv = get_or_create_feature_view()
    X, _ = fv.training_data(primary_key=True, event_time=True)
    return X
