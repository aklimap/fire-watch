"""
test_telegram.py — Verifie que les notifications Telegram fonctionnent.

Lit TELEGRAM_TOKEN et TELEGRAM_CHAT_ID depuis l'environnement (secrets GitHub),
envoie un message de test, et affiche un diagnostic clair. Le token n'est JAMAIS
affiche en entier dans les logs.
"""

import os
import requests

TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def main():
    # Verifications de base, sans reveler les valeurs
    if not TOKEN:
        print("ERREUR : le secret TELEGRAM_TOKEN est vide ou absent.")
        return
    if not CHAT_ID:
        print("ERREUR : le secret TELEGRAM_CHAT_ID est vide ou absent.")
        return
    print(f"Token detecte (longueur {len(TOKEN)}), chat_id : {CHAT_ID}")

    # 1) getMe : le token est-il valide ?
    r = requests.get(f"https://api.telegram.org/bot{TOKEN}/getMe", timeout=30)
    data = r.json()
    if not data.get("ok"):
        print(f"getMe a echoue : {data}")
        print("-> Token invalide. Regenerez-le dans BotFather et mettez a jour le secret.")
        return
    print(f"Bot OK : @{data['result'].get('username')}")

    # 2) sendMessage : l'envoi vers votre chat fonctionne-t-il ?
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": "[TEST] fire-watch : Telegram fonctionne !"},
        timeout=30,
    )
    data = r.json()
    if data.get("ok"):
        print("SUCCES : message envoye. Verifiez votre Telegram.")
    else:
        print(f"Echec de l'envoi : {data}")
        desc = data.get("description", "")
        if "chat not found" in desc:
            print("-> chat_id incorrect, ou vous n'avez pas clique 'Demarrer' sur votre bot.")
        elif "Unauthorized" in desc or data.get("error_code") == 401:
            print("-> Token refuse. Regenerez-le et mettez a jour le secret.")


if __name__ == "__main__":
    main()
