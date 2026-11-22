#!/usr/bin/env python3
"""
Joint la meteo ERA5 et la secheresse accumulee a chaque echantillon.

ERA5 ne fournit ni humidite relative ni evapotranspiration de reference :
les deux sont calculees ici, par des formules standard.

  Humidite relative : formule de Magnus, a partir de la temperature et du
  point de rosee. C'est l'humidite minimale de la journee qui compte pour
  le feu, pas la moyenne : on la derive du point de rosee minimal.

  Evapotranspiration de reference (ET0) : formule de Hargreaves, qui ne
  demande que les temperatures extremes et le rayonnement extraterrestre,
  lui-meme calculable a partir de la latitude et du jour de l'annee. Elle
  evite une variable supplementaire a telecharger, et c'est elle qui permet
  de calculer le deficit hydrique accumule.

Attributs produits par point et par date :
  conditions du jour  t_max, t_min, t_moy, hr_min, vent, pluie, et0
  secheresse cumulee  pluie a 7/30/90 j, jours sans pluie, deficit a 30/90 j
  anomalie            ecart de t_max a la normale du lieu pour ce jour

Sortie : apprentissage.csv.gz
"""
import argparse
import csv
import gzip
import math
import os
import sys
from collections import defaultdict
from datetime import date, timedelta

PLUIE_MINIMALE = 1.0   # mm : en deca, journee comptee seche
GSC = 0.0820           # constante solaire, MJ m-2 min-1

ATTRIBUTS = ["t_max", "t_min", "t_moy", "hr_min", "vent", "pluie", "et0",
             "pluie_7j", "pluie_30j", "pluie_90j", "jours_sans_pluie",
             "deficit_30j", "deficit_90j", "anomalie_t_max"]


def humidite_relative(t_c, rosee_c):
    """Formule de Magnus : humidite relative en pourcent."""
    def e(t):
        return math.exp((17.625 * t) / (243.04 + t))
    return max(0.0, min(100.0, 100.0 * e(rosee_c) / e(t_c)))


def rayonnement_extraterrestre(latitude_deg, jour_annee):
    """Ra en mm/jour equivalent, pour la formule de Hargreaves."""
    phi = math.radians(latitude_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * jour_annee / 365)
    delta = 0.409 * math.sin(2 * math.pi * jour_annee / 365 - 1.39)
    x = -math.tan(phi) * math.tan(delta)
    omega = math.acos(max(-1.0, min(1.0, x)))
    ra_mj = ((24 * 60 / math.pi) * GSC * dr
             * (omega * math.sin(phi) * math.sin(delta)
                + math.cos(phi) * math.cos(delta) * math.sin(omega)))
    return 0.408 * ra_mj


def et0_hargreaves(t_max, t_min, t_moy, latitude, jour_annee):
    ecart = max(0.0, t_max - t_min)
    ra = rayonnement_extraterrestre(latitude, jour_annee)
    return max(0.0, 0.0023 * (t_moy + 17.8) * math.sqrt(ecart) * ra)


def charger_era5(dossier, lats, lons):
    """Extrait les seuls points de grille utiles des fichiers NetCDF."""
    import numpy as np
    import xarray as xr

    noms = ["t_max", "t_min", "t_moy", "rosee_moy", "rosee_min",
            "vent_u", "vent_v", "pluie"]
    tables = {}
    coord_lat = coord_lon = None

    for nom in noms:
        chemin = os.path.join(dossier, f"{nom}.nc")
        if not os.path.exists(chemin):
            raise SystemExit(f"Fichier manquant : {chemin}. "
                             "Lancer d'abord telecharger_era5.py.")
        ds = xr.open_dataset(chemin)
        var = [v for v in ds.data_vars if ds[v].ndim >= 3][0]
        dim_lat = "latitude" if "latitude" in ds.coords else "lat"
        dim_lon = "longitude" if "longitude" in ds.coords else "lon"
        dim_t = "valid_time" if "valid_time" in ds.coords else "time"

        sel = ds[var].sel(
            {dim_lat: xr.DataArray(lats, dims="point"),
             dim_lon: xr.DataArray(lons, dims="point")}, method="nearest")
        tables[nom] = sel.values                      # (temps, point)
        if coord_lat is None:
            temps = [str(t)[:10] for t in ds[dim_t].values]
            coord_lat = sel[dim_lat].values
            coord_lon = sel[dim_lon].values
        ds.close()
        print(f"  {nom} : {tables[nom].shape}")

    return tables, temps, coord_lat, coord_lon


def construire(tables, temps, lats_grille, n_points):
    """Convertit les champs bruts en attributs quotidiens, point par point."""
    import numpy as np

    jours = [date.fromisoformat(t) for t in temps]
    doy = [j.timetuple().tm_yday for j in jours]
    n = len(jours)
    resultat = {}

    for p in range(n_points):
        lat = float(lats_grille[p])
        t_max = tables["t_max"][:, p] - 273.15
        t_min = tables["t_min"][:, p] - 273.15
        t_moy = tables["t_moy"][:, p] - 273.15
        rosee_min = tables["rosee_min"][:, p] - 273.15
        u = tables["vent_u"][:, p]
        v = tables["vent_v"][:, p]
        # kg m-2 s-1 -> mm/jour
        pluie = tables["pluie"][:, p] * 86400.0
        vent = np.sqrt(u * u + v * v) * 3.6          # m/s -> km/h

        hr_min = np.array([humidite_relative(float(a), float(b))
                           for a, b in zip(t_max, rosee_min)])
        et0 = np.array([et0_hargreaves(float(a), float(b), float(c), lat, d)
                        for a, b, c, d in zip(t_max, t_min, t_moy, doy)])

        cp = np.concatenate([[0.0], np.cumsum(pluie)])
        ce = np.concatenate([[0.0], np.cumsum(et0)])

        def fenetre(cum, i, larg):
            return float(cum[i + 1] - cum[max(0, i - larg + 1)])

        secs = np.empty(n, dtype=int)
        compteur = 0
        for i in range(n):
            secs[i] = compteur
            compteur = 0 if pluie[i] >= PLUIE_MINIMALE else compteur + 1

        normale = defaultdict(list)
        for i in range(n):
            normale[doy[i]].append(t_max[i])
        normale = {d: float(np.mean(v)) for d, v in normale.items()}

        for i, t in enumerate(temps):
            p30p, p90p = fenetre(cp, i, 30), fenetre(cp, i, 90)
            resultat[(p, t)] = {
                "t_max": round(float(t_max[i]), 2),
                "t_min": round(float(t_min[i]), 2),
                "t_moy": round(float(t_moy[i]), 2),
                "hr_min": round(float(hr_min[i]), 1),
                "vent": round(float(vent[i]), 1),
                "pluie": round(float(pluie[i]), 2),
                "et0": round(float(et0[i]), 2),
                "pluie_7j": round(fenetre(cp, i, 7), 2),
                "pluie_30j": round(p30p, 2),
                "pluie_90j": round(p90p, 2),
                "jours_sans_pluie": int(secs[i]),
                "deficit_30j": round(fenetre(ce, i, 30) - p30p, 2),
                "deficit_90j": round(fenetre(ce, i, 90) - p90p, 2),
                "anomalie_t_max": round(float(t_max[i]) - normale[doy[i]], 2),
            }
    return resultat


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--echantillons", default="echantillons.csv.gz")
    p.add_argument("--era5", default="era5")
    p.add_argument("--sortie", default="apprentissage.csv.gz")
    args = p.parse_args()

    with gzip.open(args.echantillons, "rt", encoding="utf-8") as f:
        echantillons = list(csv.DictReader(f))
    print(f"Echantillons : {len(echantillons)}")

    points = sorted({(float(e["meteo_lat"]), float(e["meteo_lon"]))
                     for e in echantillons})
    index_point = {pt: i for i, pt in enumerate(points)}
    lats = [pt[0] for pt in points]
    lons = [pt[1] for pt in points]
    print(f"Points de grille utiles : {len(points)}\n")

    tables, temps, lats_g, _ = charger_era5(args.era5, lats, lons)
    print("\nCalcul des attributs...")
    table = construire(tables, temps, lats_g, len(points))
    print(f"{len(table)} couples (point, date) prepares.\n")

    manquants = 0
    colonnes = list(echantillons[0].keys()) + ATTRIBUTS
    with gzip.open(args.sortie, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=colonnes)
        w.writeheader()
        for e in echantillons:
            cle = (index_point[(float(e["meteo_lat"]), float(e["meteo_lon"]))],
                   e["date"])
            attributs = table.get(cle)
            if attributs is None:
                manquants += 1
                attributs = {a: "" for a in ATTRIBUTS}
            ligne = dict(e)
            ligne.update(attributs)
            w.writerow(ligne)

    taille = os.path.getsize(args.sortie) / 1e6
    print(f"Ecrit : {args.sortie} ({taille:.1f} Mo)")
    if manquants:
        print(f"[!] {manquants} echantillon(s) sans meteo "
              f"({manquants/len(echantillons):.1%}) : date hors periode ERA5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
