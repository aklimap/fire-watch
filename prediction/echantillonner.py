#!/usr/bin/env python3
"""
Construit la table d'apprentissage : departs de feu et NON-departs.

C'est l'etape ou la plupart des etudes se trompent. Un modele entraine sur
les seuls departs n'apprend rien : il faut lui montrer des situations ou
aucun feu n'a demarre. La facon de choisir ces contre-exemples determine ce
que le modele apprend reellement.

Deux familles de contre-exemples, tirees en appariement :

  - temporels : MEME cellule qu'un depart, mais une autre date sans feu.
    Le lieu est tenu constant, donc seules les conditions du jour varient.
    C'est ce qui force le modele a apprendre la meteo et la secheresse.

  - spatiaux : MEME date qu'un depart, mais une autre cellule sans feu.
    La date est tenue constante, donc seul le lieu varie. C'est ce qui
    force le modele a apprendre le terrain et l'exposition.

Sans cet appariement, un tirage uniforme sur toute l'Algerie donnerait
surtout des points de plein Sahara : le modele apprendrait a reconnaitre le
desert, obtiendrait un score superbe, et serait inutile.

Domaine d'etude : les cellules ayant connu au moins un depart en 14 ans.
Ce choix restreint volontairement la question a "quand, en un lieu qui peut
bruler" plutot qu'a "ou peut-il bruler". Elargir le domaine demande une
carte d'occupation du sol (etape suivante).

Sortie : echantillons.csv.gz
"""
import argparse
import csv
import gzip
import json
import random
from collections import defaultdict
from datetime import date, timedelta

PAS_GRILLE = 0.02      # cellule ~2 km, identique a celle des departs
PAS_METEO = 0.25       # cellule meteo, resolution native d'ERA5
GARDE_JOURS = 3        # une date a moins de 3 jours d'un feu n'est pas un non-depart


def cellule(lat, lon, pas=PAS_GRILLE):
    return (round(lat / pas), round(lon / pas))


def charger_departs(chemin):
    with gzip.open(chemin, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--departs", default="departs_algerie.csv.gz")
    p.add_argument("--sortie", default="echantillons.csv.gz")
    p.add_argument("--lat-min", type=float, default=None,
                   help="restreindre au nord de cette latitude (ex. 33)")
    p.add_argument("--par-depart", type=int, default=1,
                   help="contre-exemples de chaque famille, par depart")
    p.add_argument("--graine", type=int, default=20261122)
    args = p.parse_args()

    alea = random.Random(args.graine)
    departs = charger_departs(args.departs)
    if args.lat_min is not None:
        departs = [d for d in departs if float(d["lat"]) >= args.lat_min]
    print(f"Departs retenus : {len(departs)}")

    # Domaine : cellules ayant connu au moins un depart, et leur position.
    position = {}
    feux_par_cellule = defaultdict(set)
    feux_par_date = defaultdict(set)
    for d in departs:
        c = cellule(float(d["lat"]), float(d["lon"]))
        position.setdefault(c, (float(d["lat"]), float(d["lon"])))
        j = date.fromisoformat(d["date"])
        feux_par_cellule[c].add(j)
        feux_par_date[j].add(c)

    cellules = sorted(position)
    jours = sorted({date.fromisoformat(d["date"]) for d in departs})
    debut, fin = jours[0], jours[-1]
    etendue = (fin - debut).days
    print(f"Domaine : {len(cellules)} cellules de ~2 km")
    print(f"Periode : {debut} -> {fin} ({etendue + 1} jours)\n")

    def libre(c, j):
        """Vrai si aucun depart dans cette cellule a moins de GARDE_JOURS."""
        for k in range(-GARDE_JOURS, GARDE_JOURS + 1):
            if (j + timedelta(days=k)) in feux_par_cellule[c]:
                return False
        return True

    lignes = []
    for d in departs:
        c = cellule(float(d["lat"]), float(d["lon"]))
        lignes.append({
            "cible": 1, "type": "depart",
            "date": d["date"], "lat": d["lat"], "lon": d["lon"],
            "frp_max": d["frp_max"], "duree_jours": d["duree_jours"],
        })

        # Contre-exemples temporels : meme cellule, autre date sans feu.
        for _ in range(args.par_depart):
            for _essai in range(40):
                j = debut + timedelta(days=alea.randint(0, etendue))
                if libre(c, j):
                    la, lo = position[c]
                    lignes.append({
                        "cible": 0, "type": "non_depart_temporel",
                        "date": j.isoformat(), "lat": f"{la:.5f}", "lon": f"{lo:.5f}",
                        "frp_max": "", "duree_jours": "",
                    })
                    break

        # Contre-exemples spatiaux : meme date, autre cellule sans feu.
        j = date.fromisoformat(d["date"])
        for _ in range(args.par_depart):
            for _essai in range(40):
                c2 = cellules[alea.randrange(len(cellules))]
                if c2 not in feux_par_date[j] and libre(c2, j):
                    la, lo = position[c2]
                    lignes.append({
                        "cible": 0, "type": "non_depart_spatial",
                        "date": j.isoformat(), "lat": f"{la:.5f}", "lon": f"{lo:.5f}",
                        "frp_max": "", "duree_jours": "",
                    })
                    break

    # Attributs calendaires et rattachement a la grille meteo.
    for l in lignes:
        j = date.fromisoformat(l["date"])
        la, lo = float(l["lat"]), float(l["lon"])
        cm = cellule(la, lo, PAS_METEO)
        l["jour_annee"] = j.timetuple().tm_yday
        l["mois"] = j.month
        l["jour_semaine"] = j.isoweekday()
        l["annee"] = j.year
        l["meteo_lat"] = round(cm[0] * PAS_METEO, 2)
        l["meteo_lon"] = round(cm[1] * PAS_METEO, 2)

    lignes.sort(key=lambda l: (l["date"], l["lat"], l["lon"]))
    with gzip.open(args.sortie, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0].keys()))
        w.writeheader()
        w.writerows(lignes)

    from collections import Counter
    rep = Counter(l["type"] for l in lignes)
    pos = sum(1 for l in lignes if l["cible"] == 1)
    grille = {(l["meteo_lat"], l["meteo_lon"]) for l in lignes}
    print("Echantillons :")
    for t, n in sorted(rep.items()):
        print(f"  {t:<22} {n:>7}")
    print(f"  {'TOTAL':<22} {len(lignes):>7}   (positifs {pos/len(lignes):.0%})")
    print(f"\nCellules meteo a telecharger : {len(grille)}")
    print(f"Ecrit : {args.sortie}")

    json.dump({
        "graine": args.graine,
        "lat_min": args.lat_min,
        "departs": len(departs),
        "echantillons": len(lignes),
        "repartition": dict(rep),
        "cellules_2km": len(cellules),
        "cellules_meteo": len(grille),
        "periode": {"debut": debut.isoformat(), "fin": fin.isoformat()},
        "garde_jours": GARDE_JOURS,
    }, open("resume_echantillons.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
