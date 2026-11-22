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
DEMANDES = [
    ("2m_temperature", "daily_maximum", "t_max"),
    ("2m_temperature", "daily_minimum", "t_min"),
    ("2m_temperature", "daily_mean", "t_moy"),
    ("2m_dewpoint_temperature", "daily_mean", "rosee_moy"),
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
    p.add_argument("--max-requetes", type=int, default=8,
                   help="requetes par execution (le CDS met en file d'attente)")
    args = p.parse_args()

    try:
        import cdsapi
    except ImportError:
        print("cdsapi absent : pip install 'cdsapi>=0.7.7'")
        return 1

    os.makedirs(args.dossier, exist_ok=True)
    annees = [str(a) for a in range(args.debut, args.fin + 1)]
    client = cdsapi.Client()

    a_faire = [(v, s, n) for v, s, n in DEMANDES
               if not os.path.exists(os.path.join(args.dossier, f"{n}.nc"))]
    print(f"{len(DEMANDES) - len(a_faire)} fichier(s) deja present(s), "
          f"{len(a_faire)} a telecharger.")
    print(f"Periode {annees[0]}-{annees[-1]}, emprise {EMPRISE}\n")

    obtenus, echecs = 0, []
    for variable, statistique, nom in a_faire[:args.max_requetes]:
        cible = os.path.join(args.dossier, f"{nom}.nc")
        print(f"-> {nom} ({variable}, {statistique})")
        requete = {
            "product_type": "reanalysis",
            "variable": [variable],
            "year": annees,
            "month": MOIS,
            "day": JOURS,
            "daily_statistic": statistique,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": EMPRISE,
        }
        try:
            client.retrieve(DATASET, requete).download(cible)
            taille = os.path.getsize(cible) / 1e6
            print(f"   recu : {cible} ({taille:.1f} Mo)\n")
            obtenus += 1
        except Exception as e:
            print(f"   ECHEC : {type(e).__name__}: {str(e)[:300]}\n")
            echecs.append(nom)
            if os.path.exists(cible):
                os.remove(cible)

    reste = [n for _, _, n in DEMANDES
             if not os.path.exists(os.path.join(args.dossier, f"{n}.nc"))]
    print(f"{obtenus} fichier(s) recu(s).")
    if echecs:
        print(f"Echecs : {', '.join(echecs)}")
    if reste:
        print(f"Reste a obtenir : {', '.join(reste)} -> relancer le workflow.")
        return 2
    print("Toutes les variables ERA5 sont disponibles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
