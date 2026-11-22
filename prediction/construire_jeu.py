#!/usr/bin/env python3
"""
Construit le jeu de donnees de DEPARTS de feu en Algerie, a partir de
l'archive pluriannuelle FIRMS.

Pourquoi les departs et non les detections : un meme incendie produit des
dizaines de detections sur plusieurs jours et plusieurs capteurs. Entrainer
un modele la-dessus revient a lui apprendre quels feux durent longtemps, pas
ou un feu commence. Ce script ne garde que la premiere detection de chaque
episode.

Source : archive annuelle par pays de la NASA, sans cle API ni compte.
  https://firms.modaps.eosdis.nasa.gov/data/country/zips/{capteur}_{annee}_all_countries.zip
Le serveur accepte les requetes HTTP Range : on extrait le seul fichier
Algerie de chaque archive au lieu de telecharger les ~30 Mo de tous les pays.

Sortie : departs_algerie.csv.gz  (un depart par ligne)
         resume.json             (statistiques de construction)
"""
import argparse
import csv
import gzip
import io
import json
import math
import os
import sys
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from datetime import date, datetime

BASE = "https://firms.modaps.eosdis.nasa.gov/data/country/zips"
PAYS = "Algeria"

# Couverture temporelle reelle des capteurs (source : page FIRMS Archive Download)
COUVERTURE = {
    "modis": (2000, 2025),        # MODIS Collection 6.1, depuis novembre 2000
    "viirs-snpp": (2012, 2025),   # VIIRS S-NPP 375 m, depuis janvier 2012
}

# Un episode se termine apres ce nombre de jours sans detection dans la cellule.
TROU_EPISODE_J = 3
# Taille de cellule pour regrouper les detections d'un meme feu (~2 km).
PAS_GRILLE = 0.02
# Rayon terrestre, pour les distances.
R_TERRE = 6371.0


# --------------------------------------------------------------------------- #
# Lecture d'un ZIP distant sans le telecharger en entier
# --------------------------------------------------------------------------- #
class FichierHTTP(io.RawIOBase):
    """Fichier distant adressable, via les requetes HTTP Range.

    zipfile a besoin de se deplacer dans le fichier (seek) : il lit d'abord
    l'index en fin d'archive, puis le seul membre voulu. Avec Range, on ne
    telecharge que ces quelques kilo-octets.
    """

    def __init__(self, url):
        self.url = url
        self.position = 0
        requete = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(requete, timeout=120) as r:
            self.taille = int(r.headers["Content-Length"])
            if r.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError("le serveur n'accepte pas les requetes Range")

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.position = offset
        elif whence == os.SEEK_CUR:
            self.position += offset
        else:
            self.position = self.taille + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, taille=-1):
        if taille < 0:
            taille = self.taille - self.position
        if taille == 0 or self.position >= self.taille:
            return b""
        fin = min(self.position + taille, self.taille) - 1
        requete = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{fin}"})
        with urllib.request.urlopen(requete, timeout=300) as r:
            donnees = r.read()
        self.position += len(donnees)
        return donnees


def lire_annee(capteur, annee):
    """Renvoie les lignes CSV Algerie d'une archive annuelle, ou [] si absente."""
    url = f"{BASE}/{capteur}_{annee}_all_countries.zip"
    try:
        distant = FichierHTTP(url)
    except urllib.error.HTTPError as e:
        print(f"  {capteur} {annee} : indisponible (HTTP {e.code})")
        return []
    except Exception as e:
        print(f"  {capteur} {annee} : echec ({type(e).__name__}: {e})")
        return []

    with zipfile.ZipFile(io.BufferedReader(distant, buffer_size=1 << 20)) as z:
        noms = [n for n in z.namelist() if PAYS.lower() in n.lower()]
        if not noms:
            print(f"  {capteur} {annee} : aucun fichier {PAYS} dans l'archive")
            return []
        with z.open(noms[0]) as f:
            texte = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            lignes = list(csv.DictReader(texte))
    print(f"  {capteur} {annee} : {len(lignes):>7} detections ({noms[0]})")
    return lignes


# --------------------------------------------------------------------------- #
# Filtrage
# --------------------------------------------------------------------------- #
def haversine_km(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(math.radians, (la1, lo1, la2, lo2))
    a = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R_TERRE * math.asin(math.sqrt(a))


def charger_masque(chemin):
    """Masque des sources industrielles calibre sur l'historique recent."""
    if not os.path.exists(chemin):
        print(f"[!] {chemin} absent : masque non applique.")
        return []
    d = json.load(open(chemin, encoding="utf-8"))
    defaut = d.get("rayon_km_defaut", 3.0)
    return [(s["lat"], s["lon"], s.get("rayon_km", defaut)) for s in d.get("sources", [])]


def confiance_ok(ligne):
    """VIIRS : on garde nominal et high. MODIS : seuil a 50."""
    c = str(ligne.get("confidence", "")).strip().lower()
    if c in ("l", "n", "h"):
        return c in ("n", "h")
    try:
        return float(c) >= 50
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Extraction des departs
# --------------------------------------------------------------------------- #
def extraire_departs(detections, masque):
    """Regroupe les detections par cellule, puis isole le debut de chaque episode."""
    cellules = defaultdict(list)
    rejets = defaultdict(int)

    for d in detections:
        try:
            lat = float(d["latitude"])
            lon = float(d["longitude"])
        except (KeyError, ValueError):
            rejets["coordonnees illisibles"] += 1
            continue

        # type 2 = source fixe au sol, deja signalee par la NASA dans l'archive.
        if str(d.get("type", "")).strip() == "2":
            rejets["source fixe (type NASA)"] += 1
            continue
        if not confiance_ok(d):
            rejets["confiance insuffisante"] += 1
            continue
        if any(haversine_km(lat, lon, la, lo) <= r for la, lo, r in masque):
            rejets["masque torcheres"] += 1
            continue

        try:
            jour = datetime.strptime(d["acq_date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            rejets["date illisible"] += 1
            continue

        cle = (round(lat / PAS_GRILLE), round(lon / PAS_GRILLE))
        cellules[cle].append({
            "jour": jour,
            "heure": str(d.get("acq_time", "")).zfill(4),
            "lat": lat, "lon": lon,
            "frp": float(d["frp"]) if d.get("frp") not in (None, "") else None,
            "capteur": d.get("satellite") or d.get("instrument") or "",
            "jour_nuit": d.get("daynight", ""),
        })

    departs = []
    for cle, obs in cellules.items():
        obs.sort(key=lambda o: (o["jour"], o["heure"]))
        episode = [obs[0]]
        for precedent, courant in zip(obs, obs[1:]):
            if (courant["jour"] - precedent["jour"]).days > TROU_EPISODE_J:
                departs.append(resumer(episode))
                episode = [courant]
            else:
                episode.append(courant)
        departs.append(resumer(episode))

    departs.sort(key=lambda d: (d["date"], d["heure"]))
    return departs, dict(rejets)


def resumer(episode):
    """Un depart = la premiere observation, enrichie de la suite de l'episode."""
    premiere = episode[0]
    frps = [o["frp"] for o in episode if o["frp"] is not None]
    return {
        "date": premiere["jour"].isoformat(),
        "heure": premiere["heure"],
        "lat": round(premiere["lat"], 5),
        "lon": round(premiere["lon"], 5),
        "frp_initial": premiere["frp"],
        "frp_max": round(max(frps), 1) if frps else None,
        "capteur": premiere["capteur"],
        "jour_nuit": premiere["jour_nuit"],
        "n_detections": len(episode),
        "duree_jours": (episode[-1]["jour"] - premiere["jour"]).days,
        "jour_semaine": premiere["jour"].isoweekday(),
        "mois": premiere["jour"].month,
    }


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debut", type=int, default=2012, help="premiere annee")
    p.add_argument("--fin", type=int, default=2025, help="derniere annee")
    p.add_argument("--capteurs", default="viirs-snpp,modis")
    p.add_argument("--masque", default="../sources_permanentes.json")
    p.add_argument("--sortie", default="departs_algerie.csv.gz")
    args = p.parse_args()

    masque = charger_masque(args.masque)
    print(f"Masque : {len(masque)} source(s) industrielle(s).\n")

    brutes = []
    par_source = {}
    for capteur in args.capteurs.split(","):
        capteur = capteur.strip()
        a0, a1 = COUVERTURE.get(capteur, (args.debut, args.fin))
        print(f"{capteur} :")
        n = 0
        for annee in range(max(args.debut, a0), min(args.fin, a1) + 1):
            lignes = lire_annee(capteur, annee)
            brutes.extend(lignes)
            n += len(lignes)
        par_source[capteur] = n
        print()

    if not brutes:
        print("Aucune donnee recuperee : rien a ecrire.")
        return 1

    print(f"Total detections brutes : {len(brutes)}")
    departs, rejets = extraire_departs(brutes, masque)
    print(f"Departs extraits        : {len(departs)}\n")
    print("Detections ecartees :")
    for motif, n in sorted(rejets.items(), key=lambda x: -x[1]):
        print(f"  {motif:<28} {n:>8}")

    colonnes = list(departs[0].keys())
    with gzip.open(args.sortie, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        w.writerows(departs)
    print(f"\nEcrit : {args.sortie} ({os.path.getsize(args.sortie)/1e6:.1f} Mo)")

    annees = sorted({d["date"][:4] for d in departs})
    resume = {
        "construit_le": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "periode": {"debut": departs[0]["date"], "fin": departs[-1]["date"]},
        "annees": annees,
        "detections_brutes": len(brutes),
        "detections_par_capteur": par_source,
        "departs": len(departs),
        "ecartes": rejets,
        "parametres": {
            "trou_episode_jours": TROU_EPISODE_J,
            "pas_grille_degres": PAS_GRILLE,
            "sources_masquees": len(masque),
        },
        "departs_par_annee": {a: sum(1 for d in departs if d["date"][:4] == a)
                              for a in annees},
    }
    json.dump(resume, open("resume.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("Ecrit : resume.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
