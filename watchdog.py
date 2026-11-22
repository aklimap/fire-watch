#!/usr/bin/env python3
"""
Surveillance anti-panne silencieuse d'AgroSat DZ.

Verifie l'age des fichiers de feux publies par les deux chaines
(FIRMS et MTG) et envoie une alerte Telegram si l'une d'elles a
cesse de se mettre a jour. Sans ce controle, une panne (jeton
expire, cron-job.org indisponible, identifiants EUMETSAT revoques,
changement d'API) passerait totalement inapercue.

N'utilise que la bibliotheque standard : aucune dependance.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

SOURCES = {
    "FIRMS": "https://raw.githubusercontent.com/aklimap/fire-watch/main/feux_firms.json",
    "MTG": "https://raw.githubusercontent.com/aklimap/mtg-fire/main/feux_mtg.json",
}

# Seuils en minutes. FIRMS tourne toutes les 20 min, MTG toutes les 10 min.
# On laisse une marge confortable pour absorber un retard ponctuel.
SEUILS = {"FIRMS": 90, "MTG": 60}

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def age_en_minutes(url):
    """Renvoie l'age, en minutes, du champ 'mis_a_jour' du JSON distant."""
    horodatage = int(datetime.now(timezone.utc).timestamp())
    requete = urllib.request.Request(
        f"{url}?t={horodatage}",
        headers={"Cache-Control": "no-cache", "User-Agent": "agrosat-watchdog"},
    )
    with urllib.request.urlopen(requete, timeout=30) as reponse:
        donnees = json.load(reponse)

    maj = datetime.fromisoformat(donnees["mis_a_jour"])
    if maj.tzinfo is None:
        maj = maj.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - maj).total_seconds() / 60.0


def envoyer_telegram(texte):
    if not TOKEN or not CHAT_ID:
        print("Secrets Telegram absents : alerte non envoyee.")
        return
    corps = json.dumps({"chat_id": CHAT_ID, "text": texte}).encode("utf-8")
    requete = urllib.request.Request(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data=corps,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(requete, timeout=30) as reponse:
        reponse.read()
    print("Alerte Telegram envoyee.")


def main():
    problemes = []

    for nom, url in SOURCES.items():
        seuil = SEUILS[nom]
        try:
            age = age_en_minutes(url)
        except Exception as erreur:
            problemes.append(f"{nom} : fichier illisible ({erreur})")
            print(f"{nom} : ERREUR {erreur}")
            continue

        print(f"{nom} : derniere maj il y a {age:.0f} min (seuil {seuil} min)")
        if age > seuil:
            problemes.append(
                f"{nom} : aucune mise a jour depuis {age:.0f} min (seuil {seuil} min)"
            )

    if not problemes:
        print("Les deux chaines sont a jour.")
        return

    texte = (
        "ALERTE AgroSat DZ\n"
        "La surveillance des feux est interrompue.\n\n"
        + "\n".join(f"- {p}" for p in problemes)
        + "\n\nVerifier : cron-job.org, les executions GitHub Actions, "
        "et la validite des cles EUMETSAT / FIRMS."
    )
    print(texte)
    envoyer_telegram(texte)
    sys.exit(1)


if __name__ == "__main__":
    main()
