"""
fire_alerts.py — Couche d'alerte du systeme de detection active de feux.

Prend les evenements classes par firms_watch.py et :
  1. les filtre par zone d'interet (geofencing point-dans-polygone) ;
  2. leur attribue une severite a partir de la FRP ;
  3. evite le spam (cooldown par cellule ~1 km) ;
  4. envoie une notification (Telegram, ou console en mode test).

Concu comme module enfichable : voir wire_into_firms_watch() en bas.

Dependances : requests (uniquement pour l'envoi Telegram reel)
    pip install requests

Telegram : creez un bot via @BotFather -> token ; recuperez votre chat_id via
@userinfobot. L'API est https://api.telegram.org/bot<token>/sendMessage
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")      # vide -> mode test (console)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

COOLDOWN_HOURS = 12           # ne pas re-alerter la meme cellule avant N heures
GRID_DECIMALS = 2             # 2 decimales ~ cellule de ~1 km pour l'anti-spam
ALERT_STATE_FILE = Path("alerts_sent.json")

# Seuils de severite par FRP (MW) — a calibrer sur votre terrain.
FRP_SURVEILLER = 10.0
FRP_URGENT = 50.0

# Zones d'interet : polygones [[lon, lat], ...] (fermes ou non, peu importe).
# Exemple indicatif autour des forets de Kabylie (Tizi Ouzou) — a remplacer
# par vos vrais contours (export QGIS / GeoJSON).
# Zones d'interet : polygones [[lon, lat], ...] (fermes ou non, peu importe).
ZONES_INTERET = [
    {
        "nom": "Nord Algerie",
        "polygone": [
            [-2.5, 34.0], [8.7, 34.0], [8.7, 37.1], [-2.5, 37.1],
        ],
    },
    # {"nom": "Perimetre X", "polygone": [...]},   # ligne d'exemple, peut rester
]


# --------------------------------------------------------------------------- #
# 1. Geofencing : point dans polygone (ray casting, sans dependance lourde)
# --------------------------------------------------------------------------- #
def point_in_polygon(lon, lat, polygone):
    """True si (lon, lat) est dans le polygone. Pour des contours complexes,
    preferez shapely (Point.within(Polygon))."""
    inside = False
    n = len(polygone)
    j = n - 1
    for i in range(n):
        xi, yi = polygone[i]
        xj, yj = polygone[j]
        intersecte = ((yi > lat) != (yj > lat)) and \
            (lon < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi)
        if intersecte:
            inside = not inside
        j = i
    return inside


def zones_contenant(lon, lat):
    """Liste des noms de zones d'interet contenant le point."""
    return [z["nom"] for z in ZONES_INTERET
            if point_in_polygon(lon, lat, z["polygone"])]


# --------------------------------------------------------------------------- #
# 2. Severite
# --------------------------------------------------------------------------- #
def severite(frp_max):
    if frp_max is None:
        return "INFO"
    if frp_max >= FRP_URGENT:
        return "URGENT"
    if frp_max >= FRP_SURVEILLER:
        return "SURVEILLER"
    return "INFO"


# --------------------------------------------------------------------------- #
# 3. Anti-spam (cooldown par cellule)
# --------------------------------------------------------------------------- #
def _cle_cellule(lat, lon):
    return f"{round(lat, GRID_DECIMALS)}_{round(lon, GRID_DECIMALS)}"


def _charger_envois():
    if ALERT_STATE_FILE.exists():
        return json.loads(ALERT_STATE_FILE.read_text(encoding="utf-8"))
    return {}


def _sauver_envois(d):
    ALERT_STATE_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                                encoding="utf-8")


def doit_alerter(lat, lon, envois, maintenant):
    """True si la cellule n'a pas deja ete alertee dans la fenetre de cooldown."""
    cle = _cle_cellule(lat, lon)
    dernier = envois.get(cle)
    if dernier is None:
        return True
    t = datetime.fromisoformat(dernier)
    return (maintenant - t) >= timedelta(hours=COOLDOWN_HOURS)


# --------------------------------------------------------------------------- #
# 4. Mise en forme et envoi
# --------------------------------------------------------------------------- #
def formater_message(e, zones, sev):
    maps = f"https://www.google.com/maps?q={e['lat']},{e['lon']}"
    icone = {"URGENT": "[URGENT]", "SURVEILLER": "[SURVEILLER]", "INFO": "[INFO]"}[sev]
    return (
        f"{icone} Detection feu actif\n"
        f"Zone : {', '.join(zones)}\n"
        f"Position : {e['lat']:.4f}, {e['lon']:.4f}\n"
        f"Intensite (FRP max) : {e.get('frp_max')} MW | pixels : {e['n_pixels']}\n"
        f"Capteurs : {', '.join(e['sensors'])}\n"
        f"Dernier passage : {e['last_acq']} UTC\n"
        f"Carte : {maps}"
    )


def envoyer(message):
    """Envoie via Telegram si configure, sinon affiche en console (mode test)."""
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        print("---- ALERTE (mode test, pas d'envoi) ----")
        print(message)
        print("-----------------------------------------")
        return True
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message},
                          timeout=30)
        r.raise_for_status()
        return True
    except Exception as exc:
        print(f"[!] Echec envoi Telegram : {exc}")
        return False


# --------------------------------------------------------------------------- #
# Dispatch principal
# --------------------------------------------------------------------------- #
def dispatcher(evenements):
    """evenements : liste de dicts (sortie de firms_watch.classify).
    N'alerte que les NOUVEAUX evenements situes dans une zone d'interet et
    hors cooldown. Renvoie le nombre d'alertes envoyees."""
    maintenant = datetime.now(timezone.utc)
    envois = _charger_envois()
    n_envoyees = 0

    for e in evenements:
        if e.get("status") != "NOUVEAU":
            continue
        zones = zones_contenant(e["lon"], e["lat"])
        if not zones:
            continue  # hors de toute zone surveillee
        if not doit_alerter(e["lat"], e["lon"], envois, maintenant):
            continue  # deja alerte recemment

        sev = severite(e.get("frp_max"))
        if envoyer(formater_message(e, zones, sev)):
            envois[_cle_cellule(e["lat"], e["lon"])] = maintenant.isoformat()
            n_envoyees += 1

    _sauver_envois(envois)
    return n_envoyees


# --------------------------------------------------------------------------- #
# Integration avec firms_watch.py
# --------------------------------------------------------------------------- #
def wire_into_firms_watch():
    """A la fin du main() de firms_watch.py, remplacez l'ecriture d'etat par :

        from fire_alerts import dispatcher
        n = dispatcher(classified)
        print(f"{n} alerte(s) envoyee(s).")
        save_state({"updated": now, "events": classified})
    """


# --------------------------------------------------------------------------- #
# Demo autonome (donnees fictives) : python fire_alerts.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    demo = [
        {"status": "NOUVEAU", "lat": 36.70, "lon": 4.10, "n_pixels": 6,
         "frp_max": 73.2, "sensors": ["VIIRS_NOAA20_NRT", "MODIS_NRT"],
         "last_acq": "2026-06-23 1142"},                     # dans la zone -> URGENT
        {"status": "NOUVEAU", "lat": 36.70, "lon": 4.10, "n_pixels": 6,
         "frp_max": 73.2, "sensors": ["VIIRS_NOAA20_NRT"],
         "last_acq": "2026-06-23 1230"},                     # meme cellule -> cooldown
        {"status": "NOUVEAU", "lat": 31.95, "lon": 5.83, "n_pixels": 2,
         "frp_max": 15.0, "sensors": ["MODIS_NRT"],
         "last_acq": "2026-06-23 1300"},                     # hors zone -> ignore
        {"status": "SUIVI", "lat": 36.72, "lon": 4.12, "n_pixels": 9,
         "frp_max": 90.0, "sensors": ["VIIRS_SNPP_NRT"],
         "last_acq": "2026-06-23 1305"},                     # pas NOUVEAU -> ignore
    ]
    n = dispatcher(demo)
    print(f"\n=> {n} alerte(s) envoyee(s).")
