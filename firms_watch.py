"""
firms_watch.py — Noyau d'ingestion FIRMS pour la detection active de feux.

Pipeline : requete multi-capteurs (NASA FIRMS) -> filtrage qualite ->
clustering spatial (DBSCAN haversine) -> classification NOUVEAU / SUIVI /
PERMANENT a partir d'un etat persistant et du masque des sources industrielles.

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

# L'API FIRMS raisonne en journees UTC, pas en fenetre glissante : avec
# DAY_RANGE = 1, juste apres minuit UTC la journee en cours est presque vide
# et le systeme annoncait brutalement zero feu. On demande donc deux journees
# et on filtre nous-memes sur une vraie fenetre glissante.
DAY_RANGE = 2              # journees UTC demandees a l'API (max 10)
FENETRE_DETECTION_H = 24   # on ne garde que les detections de ces dernieres heures
CLUSTER_EPS_KM = 0.75     # rayon de regroupement (~2 pixels VIIRS)
CLUSTER_MIN_SAMPLES = 1   # un pixel isole est deja un evenement candidat
NEW_EVENT_DIST_KM = 1.0   # cluster a > 1 km de tout feu connu = NOUVEAU

STATE_FILE = Path("active_events.json")
SORTIE_JSON = Path("feux_firms.json")  # fichier lu par la carte web

# Sources thermiques permanentes (torcheres, raffineries, siderurgie).
# Calibrees sur l'historique reel des detections : voir sources_permanentes.json
# pour la methode et la periode. Ces sites brulent en continu toute l'annee et
# ne doivent pas declencher d'alerte, mais restent visibles sur la carte.
SOURCES_FILE = Path("sources_permanentes.json")


def charger_sources_permanentes():
    """Renvoie [(lon, lat, rayon_km), ...] ; liste vide si le fichier manque."""
    if not SOURCES_FILE.exists():
        print("[!] sources_permanentes.json absent : aucun masque applique.")
        return []
    data = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    defaut = data.get("rayon_km_defaut", 3.0)
    sources = [(s["lon"], s["lat"], s.get("rayon_km", defaut))
               for s in data.get("sources", [])]
    print(f"Masque des sources permanentes : {len(sources)} site(s) charge(s).")
    return sources


STATIC_SOURCES = charger_sources_permanentes()

# --- Validation croisee avec la chaine geostationnaire MTG ---------------- #
# FIRMS (polaire, 375 m) et MTG (geostationnaire, ~3 km) sont deux capteurs
# independants. Un foyer vu par les deux, et persistant sur plusieurs trames
# MTG, est bien plus sur qu'une detection isolee.
URL_MTG = "https://raw.githubusercontent.com/aklimap/mtg-fire/main/feux_mtg.json"
RAYON_CROISE_KM = 5.0      # tolerance d'appariement entre les deux resolutions
FENETRE_CROISEE_H = 24.0   # alignee sur la fenetre de suivi MTG

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
    """Interroge un capteur. Renvoie (DataFrame, succes).

    succes distingue deux situations que l'ancien code confondait :
    une reponse valide sans aucun feu, et un echec de l'API. Les confondre
    conduit a publier "aucun feu" pendant une panne, ce qui est le pire
    message possible pour un systeme de detection d'incendie.
    """
    w, s, e, n = AOI_BBOX
    url = f"{BASE_URL}/{MAP_KEY}/{sensor}/{w},{s},{e},{n}/{DAY_RANGE}"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        text = r.text.strip()
        # Hors CSV, l'API renvoie un message en clair : quota depasse,
        # cle invalide, service indisponible. C'est un echec, pas un zero.
        if not text or "latitude" not in text.split("\n")[0].lower():
            print(f"[!] {sensor} : reponse non CSV -> {text[:160]!r}")
            return pd.DataFrame(), False
        df = pd.read_csv(io.StringIO(text))
        df["sensor"] = sensor
        return df, True
    except Exception as exc:  # robustesse : on n'interrompt pas le cycle
        print(f"[!] {sensor} : {exc}")
        return pd.DataFrame(), False


def fetch_all():
    """Renvoie (DataFrame concatene, nombre de capteurs interroges avec succes)."""
    resultats = [fetch_sensor(s) for s in SENSORS]
    ok = sum(1 for _, succes in resultats if succes)
    frames = [df for df, succes in resultats if succes and not df.empty]
    print(f"Capteurs interroges avec succes : {ok}/{len(SENSORS)}")
    if not frames:
        return pd.DataFrame(), ok
    return pd.concat(frames, ignore_index=True), ok


# --------------------------------------------------------------------------- #
# 2. Filtrage qualite + masque des sources permanentes
# --------------------------------------------------------------------------- #
def filtrer_recentes(df):
    """Ne garde que les detections des FENETRE_DETECTION_H dernieres heures.

    Transforme la journee UTC renvoyee par l'API en fenetre glissante : sans
    cela le nombre de feux s'effondrait a chaque minuit UTC.
    """
    if df.empty or "acq_date" not in df.columns or "acq_time" not in df.columns:
        return df
    heures = pd.to_numeric(df["acq_time"], errors="coerce").fillna(0).astype(int)
    horodatage = pd.to_datetime(
        df["acq_date"].astype(str) + " "
        + heures.map(lambda v: f"{v:04d}"),
        format="%Y-%m-%d %H%M", errors="coerce", utc=True)
    limite = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=FENETRE_DETECTION_H)
    garde = horodatage.notna() & (horodatage >= limite)
    print(f"Fenetre glissante {FENETRE_DETECTION_H} h : "
          f"{int(garde.sum())}/{len(df)} detections conservees.")
    return df[garde].copy()


def keep_confident(df):
    """Confiance : VIIRS = l/n/h (on garde n,h) ; MODIS = 0-100 (on garde >=50)."""
    conf = df["confidence"].astype(str).str.strip()
    is_viirs = conf.isin(["l", "n", "h"])
    viirs_ok = is_viirs & conf.isin(["n", "h"])
    modis_ok = (~is_viirs) & pd.to_numeric(conf, errors="coerce").fillna(0).ge(50)
    return df[viirs_ok | modis_ok].copy()


def est_permanente(lat, lon):
    """Vrai si le point tombe dans une source thermique permanente connue.

    On ne supprime pas ces detections : elles restent sur la carte avec le
    statut PERMANENT. Seule l'alerte automatique est supprimee, car un site
    industriel qui brule toute l'annee n'est pas un depart de feu.
    """
    for lon_s, lat_s, rayon in STATIC_SOURCES:
        if haversine_km(lat, lon, lat_s, lon_s) <= rayon:
            return True
    return False


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
    # Les sources permanentes sont exclues de la reference : sinon un vrai
    # depart de feu a cote d'une raffinerie serait classe SUIVI, donc sans alerte.
    known = [k for k in state.get("events", []) if k.get("status") != "PERMANENT"]
    out = []
    for _, e in events_df.iterrows():
        if est_permanente(e["lat"], e["lon"]):
            rec = e.to_dict()
            rec["status"] = "PERMANENT"
            out.append(rec)
            continue
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
# 5. Validation croisee FIRMS x MTG
# --------------------------------------------------------------------------- #
def charger_foyers_mtg():
    """Recupere les foyers suivis par la chaine MTG. Renvoie [] si indisponible."""
    try:
        horodatage = int(datetime.now(timezone.utc).timestamp())
        r = requests.get(f"{URL_MTG}?t={horodatage}", timeout=30)
        r.raise_for_status()
        foyers = r.json().get("feux", [])
        print(f"Validation croisee : {len(foyers)} foyer(s) MTG charge(s).")
        return foyers
    except Exception as exc:
        print(f"[!] MTG indisponible pour la validation croisee : {exc}")
        return []


def _age_heures(last_acq):
    """Age d'une detection FIRMS ('2026-11-21 1301'), en heures."""
    try:
        jour, hhmm = last_acq.split()
        t = datetime.strptime(f"{jour} {int(hhmm):04d}", "%Y-%m-%d %H%M")
        return (datetime.now(timezone.utc) - t.replace(tzinfo=timezone.utc)
                ).total_seconds() / 3600
    except Exception:
        return None


def valider_par_mtg(events, foyers_mtg):
    """Attribue un niveau de confiance a chaque evenement FIRMS.

    HAUTE        : vu par FIRMS ET par MTG avec persistance temporelle.
                   Deux capteurs independants plus la persistance : c'est un feu.
    MOYENNE      : vu par FIRMS seul, ou par MTG sur une seule trame.
    NON_EVALUEE  : detection FIRMS trop ancienne pour la fenetre MTG (6 h),
                   ou chaine MTG indisponible. L'absence de croisement n'est
                   alors pas une information, et ne doit pas etre lue comme telle.
    """
    for e in events:
        if e.get("status") == "PERMANENT":
            e["confiance"] = "NON_EVALUEE"
            e["mtg"] = None
            continue

        age = _age_heures(e.get("last_acq", ""))
        if not foyers_mtg or age is None or age > FENETRE_CROISEE_H:
            e["confiance"] = "NON_EVALUEE"
            e["mtg"] = None
            continue

        proche = None
        for m in foyers_mtg:
            if not isinstance(m.get("lat"), (int, float)):
                continue
            if haversine_km(e["lat"], e["lon"], m["lat"], m["lon"]) <= RAYON_CROISE_KM:
                if proche is None or m.get("n_trames", 0) > proche.get("n_trames", 0):
                    proche = m

        if proche is None:
            e["confiance"] = "MOYENNE"
            e["mtg"] = None
        else:
            e["mtg"] = {"n_trames": proche.get("n_trames"),
                        "duree_min": proche.get("duree_min"),
                        "statut": proche.get("statut")}
            e["confiance"] = "HAUTE" if proche.get("statut") == "CONFIRME" else "MOYENNE"
    return events


# --------------------------------------------------------------------------- #
# Sortie pour la carte web (feux_firms.json)
# --------------------------------------------------------------------------- #
def ecrire_json_carte(classified, now, capteurs_ok=None):
    """Ecrit les feux dans feux_firms.json (meme structure que MTG) pour la carte."""
    feux = [
        {
            "lat": e["lat"],
            "lon": e["lon"],
            "frp_max": e.get("frp_max"),
            "n_pixels": e.get("n_pixels"),
            "statut": e.get("status"),
            "capteurs": e.get("sensors", []),
            "derniere_detection": e.get("last_acq"),
            "confiance": e.get("confiance"),
            "mtg": e.get("mtg"),
        }
        for e in classified
    ]
    data = {
        "source": "FIRMS",
        "mis_a_jour": now,
        "nombre": len(feux),
        "capteurs_interroges": capteurs_ok,
        "capteurs_total": len(SENSORS),
        "collecte": ("OK" if capteurs_ok == len(SENSORS) else "DEGRADEE"),
        "feux": feux,
    }
    SORTIE_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"Ecrit dans {SORTIE_JSON} ({len(feux)} feux).")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main():
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw, capteurs_ok = fetch_all()

    if capteurs_ok == 0:
        # Aucun capteur n'a repondu : on ne sait rien. Ecraser le fichier avec
        # zero feu afficherait une carte vide et rassurerait a tort, alors que
        # des incendies peuvent etre en cours. On conserve donc la derniere
        # publication : son horodatage vieillit et le watchdog finit par alerter.
        print(f"[{now}] ECHEC : aucun capteur FIRMS n'a repondu. "
              f"Fichier precedent conserve, aucune publication.")
        raise SystemExit(1)

    if raw.empty:
        print(f"[{now}] Aucune detection sur l'AOI "
              f"({capteurs_ok}/{len(SENSORS)} capteurs interroges).")
        ecrire_json_carte([], now, capteurs_ok)
        return

    df = keep_confident(filtrer_recentes(raw))
    events_df = cluster_events(df)
    state = load_state()
    classified = classify(events_df, state)
    classified = valider_par_mtg(classified, charger_foyers_mtg())

    new_events = [e for e in classified if e["status"] == "NOUVEAU"]
    permanents = [e for e in classified if e["status"] == "PERMANENT"]
    print(f"[{now}] {len(classified)} evenement(s) : {len(new_events)} NOUVEAU(X), "
          f"{len(permanents)} source(s) permanente(s) ecartee(s) des alertes.")
    hautes = [e for e in classified if e.get("confiance") == "HAUTE"]
    if hautes:
        print(f"[{now}] {len(hautes)} evenement(s) confirme(s) par les deux sources.")
    for e in classified:
        flag = ">>>" if e["status"] == "NOUVEAU" else "   "
        print(f"{flag} {e['status']:7s} {e['lat']:.4f},{e['lon']:.4f} "
              f"| pixels={e['n_pixels']} FRP_max={e['frp_max']} "
              f"| {','.join(e['sensors'])} | {e['last_acq']}")

    # Envoi des alertes (geofence + Telegram), puis persistance de l'etat.
    from fire_alerts import dispatcher
    n = dispatcher(classified)
    print(f"[{now}] {n} alerte(s) envoyee(s).")

    ecrire_json_carte(classified, now, capteurs_ok)
    save_state({"updated": now, "events": classified})


if __name__ == "__main__":
    main()
