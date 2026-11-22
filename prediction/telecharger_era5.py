#!/usr/bin/env python3
"""
Telecharge la meteo ERA5 pour toute l'Algerie depuis le Climate Data Store.

Pourquoi ce script remplace joindre_meteo.py : Open-Meteo facture une
requete par point, et son quota gratuit est pondere par (jours x variables).
Il saturait des la deuxieme cellule. Le CDS raisonne autrement : on demande
une grille entiere sur toute la periode en une seule requete par variable.
Sept requetes suffisent la ou il en fallait 435, et on recupere la pleine
resolution de 0,25 degre sur les 14 annees au lieu de 0,5 degre sur sept.

Variables demandees au jeu derived-era5-single-levels-daily-statistics :
  2m_temperature                 maximum, minimum et moyenne
  2m_dewpoint_temperature        moyenne, pour en deduire l'humidite
  10m_u/v_component_of_wind      moyenne, pour en deduire le vent
  mean_total_precipitation_rate  moyenne, convertie en mm par jour

L'humidite relative est calculee a partir de la temperature et du point de
rosee par la formule de Magnus, l'ERA5 ne la fournissant pas directement.

Le CDS met les requetes en file : chacune peut prendre de quelques minutes
a plus d'une heure. Le script enregistre chaque fichier des sa reception et
saute ceux deja presents, donc une execution interrompue reprend sans
retelecharger.

Sortie : era5/<variable>_<statistique>.nc
"""
import argparse
import os
import sys

DATASET = "derived-era5-single-levels-daily-statistics"

# (variable CDS, statistique, nom court en sortie)
# La temperature moyenne est deduite de (max + min) / 2 plutot que
# telechargee : une requete de moins, pour une difference negligeable dans
# la formule de Hargreaves.
DEMANDES = [
    ("2m_temperature", "daily_maximum", "t_max"),
    ("2m_temperature", "daily_minimum", "t_min"),
    ("2m_dewpoint_temperature", "daily_minimum", "rosee_min"),
    ("10m_u_component_of_wind", "daily_mean", "vent_u"),
    ("10m_v_component_of_wind", "daily_mean", "vent_v"),
    ("mean_total_precipitation_rate", "daily_mean", "pluie"),
]

# Emprise de l'Algerie et de ses marges, au format CDS : nord, ouest, sud, est.
EMPRISE = [38.0, -9.0, 18.0, 12.5]

MOIS = [f"{m:02d}" for m in range(1, 13)]
JOURS = [f"{j:02d}" for j in range(1, 32)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--debut", type=int, default=2012)
    p.add_argument("--fin", type=int, default=2025)
    p.add_argument("--dossier", default="era5")
    p.add_argument("--max-requetes", type=int, default=200,
                   help="requetes par execution (le CDS met en file d'attente)")
    p.add_argument("--annees-par-requete", type=int, default=4,
                   help="taille du bloc d'annees, reduite automatiquement si le "
                        "CDS juge la requete trop grande")
    args = p.parse_args()

    try:
        import cdsapi
    except ImportError:
        print("cdsapi absent : pip install 'cdsapi>=0.7.7'")
        return 1

    os.makedirs(args.dossier, exist_ok=True)
    client = cdsapi.Client()

    def blocs(taille):
        """Decoupe la periode en blocs d'annees consecutives."""
        a = args.debut
        while a <= args.fin:
            b = min(a + taille - 1, args.fin)
            yield a, b
            a = b + 1

    def chemin(nom, a, b):
        return os.path.join(args.dossier, f"{nom}_{a}_{b}.nc")

    def demander(variable, statistique, nom, a, b, taille):
        """Telecharge un bloc, en le coupant en deux si le CDS le refuse."""
        cible = chemin(nom, a, b)
        if os.path.exists(cible) and os.path.getsize(cible) > 0:
            return True, 0
        requete = {
            "product_type": "reanalysis",
            "variable": [variable],
            "year": [str(x) for x in range(a, b + 1)],
            "month": MOIS, "day": JOURS,
            "daily_statistic": statistique,
            "time_zone": "utc+00:00", "frequency": "1_hourly",
            "area": EMPRISE,
        }
        print(f"-> {nom} {a}-{b}")
        try:
            client.retrieve(DATASET, requete).download(cible)
            print(f"   recu ({os.path.getsize(cible)/1e6:.1f} Mo)")
            return True, 1
        except Exception as e:
            if os.path.exists(cible):
                os.remove(cible)
            corps = ""
            rep = getattr(e, "response", None)
            if rep is not None:
                try:
                    corps = rep.text
                except Exception:
                    pass
            trop_grand = "cost limits" in corps or "too large" in corps
            if trop_grand and a < b:
                milieu = a + (b - a) // 2
                print(f"   bloc trop grand, coupe en {a}-{milieu} et {milieu+1}-{b}")
                ok1, n1 = demander(variable, statistique, nom, a, milieu, taille)
                ok2, n2 = demander(variable, statistique, nom, milieu + 1, b, taille)
                return ok1 and ok2, n1 + n2
            print(f"   ECHEC : {type(e).__name__}: {str(e)[:200]}")
            if corps:
                print(f"   Reponse du serveur : {corps[:600]}")
            return False, 0

    print(f"Periode {args.debut}-{args.fin}, emprise {EMPRISE}")
    print(f"{len(DEMANDES)} variables, blocs de {args.annees_par_requete} an(s)\n")

    recus, echecs = 0, []
    for variable, statistique, nom in DEMANDES:
        for a, b in blocs(args.annees_par_requete):
            if recus >= args.max_requetes:
                print("\nLimite de requetes atteinte pour cette execution.")
                return 2
            ok, n = demander(variable, statistique, nom, a, b,
                             args.annees_par_requete)
            recus += n
            if not ok:
                echecs.append(f"{nom} {a}-{b}")

    print(f"\n{recus} fichier(s) recu(s) cette execution.")
    if echecs:
        print(f"Echecs : {', '.join(echecs)}")
        return 2
    print("Toutes les variables ERA5 sont disponibles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
