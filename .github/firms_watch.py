"""
firms_watch.py — Noyau d'ingestion FIRMS pour la detection active de feux.

Pipeline : requete multi-capteurs (NASA FIRMS) -> filtrage qualite ->
masque des sources thermiques permanentes (torcheres) -> clustering spatial
(DBSCAN haversine) -> classification NOUVEAU / SUIVI a partir d'un etat persistant.

AOI par defaut : Algerie. Concu pour tourner en tache planifiee (cron / Cloud Function).
L'etat est un simple JSON ; remplacez load_state/save_state par Firebase en production.

Dependances : pandas, numpy, requests, scikit-learn
    pip install pandas numpy requests scikit-learn

MAP_KEY gratuit : https://firms.modaps.eosdis.nasa.gov/api/map_key/
"""

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import DBSCAN

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")  # defini dans les secrets GitHub

# Bounding box (west, south, east, north). Algerie ~ (-8.7, 18.9, 12.0, 37.1)
AOI_BBOX = (-8.7, 18.9, 12.0, 37.1)

# Capteurs : VIIRS 375 m en priorite (meilleure resolution), MODIS en complement.
SENSORS = ["VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"]

DAY_RANGE = 1              # 1 = dernieres 24 h (max 10)
CLUSTER_EPS_KM = 0.75     # rayon de regroupement (~2 pixels VIIRS)
CLUSTER_MIN_SAMPLES = 1   # un pixel isole est deja un evenement candidat
NEW_EVENT_DIST_KM = 1.0   # cluster a > 1 km de tout feu connu = NOUVEAU

STATE_FILE = Path("active_events.json")

# Sources thermiques permanentes a exclure : torcheres, sites industriels.
# Format : (lon, lat, rayon_km). A calibrer avec la base de torcheres
# VIIRS Nightfire (Earth Observation Group). Exemples indicatifs a verifier :
STATIC_SOURCES = [
    # (5.83, 31.95, 4.0),   # zone Hassi Messaoud (exemple - a calibrer)
    # (3.20, 32.93, 4.0),   # zone Hassi R'Mel    (exemple - a calibrer)
]

EARTH_R_KM = 6371.0
BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


# --------------------------------------------------------------------------- #
# Outils geo
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    """Distance haversine (km), vectorisable via numpy."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- #
# 1. Recuperation
# --------------------------------------------------------------------------- #
def fetch_sensor(sensor):
    """Renvoie un DataFrame des detections d'un capteur sur l'AOI, ou vide."""
    w, s, e, n = AOI_BBOX
    url = f"{BASE_URL}/{MAP_KEY}/{sensor}/{w},{s},{e},{n}/{DAY_RANGE}"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        text = r.text.strip()
        # L'API renvoie un message texte (pas un CSV) en cas d'erreur/zone vide.
        if not text or "latitude" not in text.split("\n")[0].lower():
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(text))
        df["sensor"] = sensor
        return df
    except Exception as exc:  # robustesse : on n'interrompt pas le cycle
        print(f"[!] {sensor}: {exc}")
        return pd.DataFrame()


def fetch_all():
    frames = [fetch_sensor(s) for s in SENSORS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2. Filtrage qualite + masque des sources permanentes
# --------------------------------------------------------------------------- #
def keep_confident(df):
    """Confiance : VIIRS = l/n/h (on garde n,h) ; MODIS = 0-100 (on garde >=50)."""
    conf = df["confidence"].astype(str).str.strip()
    is_viirs = conf.isin(["l", "n", "h"])
    viirs_ok = is_viirs & conf.isin(["n", "h"])
    modis_ok = (~is_viirs) & pd.to_numeric(conf, errors="coerce").fillna(0).ge(50)
    return df[viirs_ok | modis_ok].copy()


def drop_static_sources(df):
    """Retire les detections proches d'une source thermique permanente connue."""
    if not STATIC_SOURCES or df.empty:
        return df
    mask = np.zeros(len(df), dtype=bool)
    for lon, lat, radius in STATIC_SOURCES:
        d = haversine_km(df["latitude"].values, df["longitude"].values, lat, lon)
        mask |= d <= radius
    return df[~mask].copy()


# --------------------------------------------------------------------------- #
# 3. Clustering : pixels bruts -> evenements
# --------------------------------------------------------------------------- #
def cluster_events(df):
    """Regroupe les pixels en evenements (DBSCAN haversine) et agrege."""
    if df.empty:
        return pd.DataFrame()
    coords = np.radians(df[["latitude", "longitude"]].values)
    eps = CLUSTER_EPS_KM / EARTH_R_KM  # eps en radians pour la metrique haversine
    labels = DBSCAN(eps=eps, min_samples=CLUSTER_MIN_SAMPLES,
                    metric="haversine").fit_predict(coords)
    df = df.assign(cluster=labels)

    frp_col = "frp" if "frp" in df.columns else None
    events = []
    for cid, g in df.groupby("cluster"):
        events.append({
            "lat": round(float(g["latitude"].mean()), 5),
            "lon": round(float(g["longitude"].mean()), 5),
            "n_pixels": int(len(g)),
            "frp_max": round(float(g[frp_col].max()), 1) if frp_col else None,
            "frp_sum": round(float(g[frp_col].sum()), 1) if frp_col else None,
            "sensors": sorted(g["sensor"].unique().tolist()),
            "last_acq": f"{g['acq_date'].max()} {int(g['acq_time'].max()):04d}",
        })
    # Tri par intensite (FRP cumulee) decroissante
    events.sort(key=lambda e: (e["frp_sum"] or 0), reverse=True)
    return pd.DataFrame(events)


# --------------------------------------------------------------------------- #
# 4. Etat persistant + classification NOUVEAU / SUIVI
# --------------------------------------------------------------------------- #
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"events": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def classify(events_df, state):
    """Marque chaque evenement NOUVEAU ou SUIVI selon la distance aux feux connus."""
    known = state.get("events", [])
    out = []
    for _, e in events_df.iterrows():
        status = "NOUVEAU"
        if known:
            dists = [haversine_km(e["lat"], e["lon"], k["lat"], k["lon"])
                     for k in known]
            if min(dists) <= NEW_EVENT_DIST_KM:
                status = "SUIVI"
        rec = e.to_dict()
        rec["status"] = status
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main():
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw = fetch_all()
    if raw.empty:
        print(f"[{now}] Aucune detection sur l'AOI.")
        return

    df = drop_static_sources(keep_confident(raw))
    events_df = cluster_events(df)
    state = load_state()
    classified = classify(events_df, state)

    new_events = [e for e in classified if e["status"] == "NOUVEAU"]
    print(f"[{now}] {len(classified)} evenement(s), dont {len(new_events)} NOUVEAU(X).")
    for e in classified:
        flag = ">>>" if e["status"] == "NOUVEAU" else "   "
        print(f"{flag} {e['status']:7s} {e['lat']:.4f},{e['lon']:.4f} "
              f"| pixels={e['n_pixels']} FRP_max={e['frp_max']} "
              f"| {','.join(e['sensors'])} | {e['last_acq']}")

    # Envoi des alertes (geofence + Telegram), puis persistance de l'etat.
    from fire_alerts import dispatcher
    n = dispatcher(classified)
    print(f"[{now}] {n} alerte(s) envoyee(s).")

    save_state({"updated": now, "events": classified})


if __name__ == "__main__":
    main()
