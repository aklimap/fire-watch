#!/usr/bin/env python3
"""
Joint les conditions meteorologiques et la secheresse accumulee a chaque
echantillon, depuis l'archive ERA5 servie par Open-Meteo (libre, sans cle).

Le jour du feu ne suffit pas a expliquer un depart : ce qui compte est
l'etat du combustible, qui resulte de semaines de deficit hydrique. Le
script calcule donc, pour chaque point :

  - les conditions du jour : temperature, humidite, vent, rayonnement ;
  - la secheresse accumulee : pluie cumulee a 7, 30 et 90 jours, jours
    consecutifs sans pluie, et deficit hydrique (evapotranspiration moins
    precipitations) a 30 et 90 jours. C'est l'equivalent du Drought Code
    du systeme canadien : la chaleur et la secheresse qui s'accumulent ;
  - les anomalies : ecart a la normale du lieu pour ce jour de l'annee.
    Trente degres n'ont pas le meme sens en janvier et en aout.

Une seule requete par cellule de 0,25 degre couvre les 14 annees. Le script
est reprenable : les cellules deja traitees sont memorisees, donc une
interruption ou une limite de debit ne fait pas tout recommencer.

Sortie : apprentissage.csv.gz  (une ligne par echantillon, prete a entrainer)
"""
import argparse
import csv
import gzip
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

API = "https://archive-api.open-meteo.com/v1/archive"
# Le quota gratuit d'Open-Meteo est pondere par (jours x variables) : une
# requete de 14 ans sur 10 variables suffit a declencher un HTTP 429 des la
# deuxieme cellule. On s'en tient aux six variables qui portent le signal.
VARIABLES = [
    "temperature_2m_max", "temperature_2m_mean",
    "relative_humidity_2m_min",
    "wind_speed_10m_max",
    "precipitation_sum", "et0_fao_evapotranspiration",
]
PLUIE_MINIMALE = 1.0      # mm : en deca, la journee compte comme seche
ETAT = "meteo_fait.json"  # cellules deja traitees


def telecharger(lat, lon, debut, fin, essais=4):
    params = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": debut, "end_date": fin,
        "daily": ",".join(VARIABLES), "timezone": "UTC",
    })
    attente = 5
    for essai in range(essais):
        try:
            with urllib.request.urlopen(f"{API}?{params}", timeout=180) as r:
                return json.load(r)["daily"]
        except urllib.error.HTTPError as e:
            corps = e.read()[:300].decode("utf-8", "replace")
            if e.code == 429:
                print(f"    limite de debit, pause {attente}s")
                time.sleep(attente); attente *= 2; continue
            raise RuntimeError(f"HTTP {e.code} : {corps}")
        except Exception as e:
            if essai == essais - 1:
                raise
            print(f"    {type(e).__name__}, nouvelle tentative dans {attente}s")
            time.sleep(attente); attente *= 2
    raise RuntimeError("echec apres plusieurs tentatives")


def derivees(serie):
    """Transforme la serie quotidienne en attributs par date."""
    jours = serie["time"]
    col = {v: serie[v] for v in VARIABLES}
    n = len(jours)

    def val(v, i):
        x = col[v][i]
        return None if x is None else float(x)

    # Normale par jour de l'annee, pour exprimer les anomalies.
    par_doy = defaultdict(list)
    for i, j in enumerate(jours):
        t = val("temperature_2m_max", i)
        if t is not None:
            par_doy[date.fromisoformat(j).timetuple().tm_yday].append(t)
    normale = {d: statistics.fmean(v) for d, v in par_doy.items() if v}

    # Cumuls glissants, calcules en un seul passage.
    cumul_pluie = [0.0] * (n + 1)
    cumul_et0 = [0.0] * (n + 1)
    for i in range(n):
        cumul_pluie[i+1] = cumul_pluie[i] + (val("precipitation_sum", i) or 0.0)
        cumul_et0[i+1] = cumul_et0[i] + (val("et0_fao_evapotranspiration", i) or 0.0)

    def somme(tab, i, fenetre):
        a = max(0, i - fenetre + 1)
        return round(tab[i+1] - tab[a], 2)

    # Jours consecutifs sans pluie significative, a la veille du jour i.
    secs = [0] * n
    compteur = 0
    for i in range(n):
        secs[i] = compteur
        compteur = 0 if (val("precipitation_sum", i) or 0.0) >= PLUIE_MINIMALE else compteur + 1

    sortie = {}
    for i, j in enumerate(jours):
        tmax = val("temperature_2m_max", i)
        doy = date.fromisoformat(j).timetuple().tm_yday
        sortie[j] = {
            "t_max": tmax,
            "t_moy": val("temperature_2m_mean", i),
            "hr_min": val("relative_humidity_2m_min", i),
            "vent_max": val("wind_speed_10m_max", i),
            "pluie": val("precipitation_sum", i),
            "et0": val("et0_fao_evapotranspiration", i),
            "pluie_7j": somme(cumul_pluie, i, 7),
            "pluie_30j": somme(cumul_pluie, i, 30),
            "pluie_90j": somme(cumul_pluie, i, 90),
            "jours_sans_pluie": secs[i],
            "deficit_30j": round(somme(cumul_et0, i, 30) - somme(cumul_pluie, i, 30), 2),
            "deficit_90j": round(somme(cumul_et0, i, 90) - somme(cumul_pluie, i, 90), 2),
            "anomalie_t_max": (round(tmax - normale[doy], 2)
                               if tmax is not None and doy in normale else None),
        }
    return sortie


ATTRIBUTS = ["t_max", "t_moy", "hr_min", "vent_max", "pluie", "et0",
             "pluie_7j", "pluie_30j", "pluie_90j", "jours_sans_pluie",
             "deficit_30j", "deficit_90j", "anomalie_t_max"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--echantillons", default="echantillons.csv.gz")
    p.add_argument("--sortie", default="apprentissage.csv.gz")
    p.add_argument("--max-cellules", type=int, default=500,
                   help="cellules traitees par execution (reprise possible)")
    p.add_argument("--pause", type=float, default=8.0, help="secondes entre requetes")
    p.add_argument("--annee-min", type=int, default=None,
                   help="ne garder que les echantillons de cette annee et apres")
    args = p.parse_args()

    with gzip.open(args.echantillons, "rt", encoding="utf-8") as f:
        echantillons = list(csv.DictReader(f))
    if args.annee_min:
        echantillons = [e for e in echantillons
                        if int(e["date"][:4]) >= args.annee_min]
    print(f"Echantillons : {len(echantillons)}")

    par_cellule = defaultdict(list)
    for e in echantillons:
        par_cellule[(e["meteo_lat"], e["meteo_lon"])].append(e)

    # Marge de 90 jours avant le premier echantillon, pour les cumuls.
    dates = [e["date"] for e in echantillons]
    debut = (date.fromisoformat(min(dates)) - timedelta(days=95)).isoformat()
    fin = max(dates)
    print(f"Serie demandee : {debut} -> {fin}")

    faites = set()
    if os.path.exists(ETAT):
        faites = {tuple(c) for c in json.load(open(ETAT))["cellules"]}
    restantes = [c for c in sorted(par_cellule) if c not in faites]
    print(f"Cellules : {len(par_cellule)} au total, {len(faites)} deja faites, "
          f"{len(restantes)} restantes\n")

    mode = "at" if os.path.exists(args.sortie) and faites else "wt"
    colonnes = list(echantillons[0].keys()) + ATTRIBUTS
    sortie = gzip.open(args.sortie, mode, newline="", encoding="utf-8")
    w = csv.DictWriter(sortie, fieldnames=colonnes)
    if mode == "wt":
        w.writeheader()

    traitees, ecrites = 0, 0
    try:
        for (la, lo) in restantes[:args.max_cellules]:
            serie = telecharger(float(la), float(lo), debut, fin)
            table = derivees(serie)
            for e in par_cellule[(la, lo)]:
                ligne = dict(e)
                ligne.update(table.get(e["date"], {a: None for a in ATTRIBUTS}))
                w.writerow(ligne)
                ecrites += 1
            faites.add((la, lo))
            traitees += 1
            if traitees % 25 == 0:
                print(f"  {traitees}/{min(len(restantes), args.max_cellules)} cellules, "
                      f"{ecrites} lignes")
            time.sleep(args.pause)
    except Exception as e:
        print(f"\n[!] Interrompu : {type(e).__name__}: {e}")
        print("    Progression conservee, relancer le workflow pour continuer.")
    finally:
        sortie.close()
        json.dump({"cellules": sorted(list(c) for c in faites),
                   "total": len(par_cellule)},
                  open(ETAT, "w"), indent=0)

    reste = len(par_cellule) - len(faites)
    print(f"\n{traitees} cellule(s) traitee(s), {ecrites} ligne(s) ecrite(s).")
    print(f"Reste {reste} cellule(s)." if reste else "\nJeu d'apprentissage complet.")
    return 0 if reste == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
