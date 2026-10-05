"""
basics_eval.py — test de RÉGRESSION sur des connaissances de base (jours, contraires, capitales, planètes…).

Ce n'est PAS un test de généralisation : ces questions figurent dans la batterie des 337 et le synthétique en apprend
une partie. Il sert à détecter qu'un nouveau SFT a PERDU ce que le précédent savait (cas du v3.1 : « 12 jours dans une
semaine », « le contraire de grand est grand »). Réponse jugée bonne si elle contient l'un des mots attendus
(insensible aux accents et à la casse). Décodage glouton.

Usage : python basics_eval.py --ckpt checkpoints/sft_A/final.pt
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Dict, List, Tuple

BASICS: List[Tuple[str, List[str]]] = [
    ("Combien de jours y a-t-il dans une semaine ?", ["sept", "7"]),
    ("Combien de mois y a-t-il dans une année ?", ["douze", "12"]),
    ("Combien d'heures y a-t-il dans une journée ?", ["24", "vingt-quatre", "vingt quatre"]),
    ("Combien de minutes y a-t-il dans une heure ?", ["soixante", "60"]),
    ("Combien de secondes y a-t-il dans une minute ?", ["soixante", "60"]),
    ("Quel jour vient après vendredi ?", ["samedi"]),
    ("Quel jour vient après le mardi ?", ["mercredi"]),
    ("Si aujourd'hui est lundi, quel jour sera demain ?", ["mardi"]),
    ("Quel est le contraire de grand ?", ["petit"]),
    ("Quel est le contraire de rapide ?", ["lent"]),
    ("Quel est le contraire de difficile ?", ["facile"]),
    ("Quel est le synonyme de heureux ?", ["joyeux", "content", "ravi"]),
    ("Quel est le féminin de acteur ?", ["actrice"]),
    ("Quelle est la capitale du Cameroun ?", ["yaounde"]),
    ("Quelle est la capitale de la France ?", ["paris"]),
    ("Quelle est la capitale du Nigeria ?", ["abuja"]),
    ("Quelle est la capitale du Tchad ?", ["djamena"]),
    ("Quelle est la capitale du Gabon ?", ["libreville"]),
    ("Quelle est la capitale du Sénégal ?", ["dakar"]),
    ("Quelle est la capitale du Ghana ?", ["accra"]),
    ("Quelle est la capitale du Kenya ?", ["nairobi"]),
    ("Quelle est la capitale de l'Égypte ?", ["caire"]),
    ("Quelle est la capitale de la République centrafricaine ?", ["bangui"]),
    ("Quelle est la capitale de la Côte d'Ivoire ?", ["yamoussoukro"]),
    ("Combien y a-t-il de planètes dans le système solaire ?", ["huit", "8"]),
    ("Quelle est la planète la plus proche du Soleil ?", ["mercure"]),
    ("Quelle est la plus grande planète du système solaire ?", ["jupiter"]),
    ("Quelle est la planète surnommée la planète rouge ?", ["mars"]),
    ("Quel est le plus long fleuve d'Afrique ?", ["nil"]),
    ("Quel est le plus grand océan du monde ?", ["pacifique"]),
    ("Si tous les chats sont des animaux et que Mimi est un chat, que peut-on dire de Mimi ?", ["animal"]),
    ("Paul est plus âgé que Jean. Jean est plus âgé que Marc. Qui est le plus jeune ?", ["marc"]),
    ("Quel est l'intrus entre chien, chat, cheval et voiture ?", ["voiture"]),
    ("Quel est l'intrus entre pomme, banane, carotte et orange ?", ["carotte"]),
    ("Corrige cette phrase : Les enfant joue dehors.", ["enfants jouent"]),
    ("Corrige cette phrase : Nous somme allé au marché.", ["sommes alles", "sommes allés"]),
]


def _fold(text: str) -> str:
    """minuscules, sans accents, apostrophes normalisées."""
    t = unicodedata.normalize("NFKD", text.lower().replace("’", "'").replace("œ", "oe").replace("æ", "ae"))
    return "".join(c for c in t if not unicodedata.combining(c))


def is_correct(reply: str, accepted: List[str]) -> bool:
    """Mot attendu présent comme MOT entier : « 7 » ne valide pas « 17 », « mars » ne valide pas « marseille »."""
    r = _fold(reply)
    return any(re.search(r"(?<![a-z0-9])" + re.escape(_fold(a)) + r"(?![a-z0-9])", r) for a in accepted)


def main():
    p = argparse.ArgumentParser(description="MiniLLM — connaissances de base (test de régression)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    from evaluate import chat_reply, load_model     # même API que skills_eval.py (déjà validée sur Colab)

    model, tok, _ = load_model(a.ckpt, device=a.device)
    ok, failed = 0, []
    for q, accepted in BASICS:
        reply = chat_reply(model, tok, [{"role": "user", "content": q}], max_new_tokens=60,
                           temperature=0.0, repetition_penalty=1.0)
        if is_correct(reply, accepted):
            ok += 1
        else:
            failed.append((q, reply))
    for q, reply in failed[:8]:
        print(f"  ✗ {q}\n      obtenu : {reply[:90]}")
    res: Dict[str, float] = {"n": len(BASICS), "correct": ok, "basics": round(ok / len(BASICS), 3)}
    print(f"basics : {ok}/{len(BASICS)} = {res['basics'] * 100:.1f} %")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
