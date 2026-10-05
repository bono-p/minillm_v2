"""
knowledge_eval.py — test de CONNAISSANCES plus fin que basics_eval (36 questions), tenues hors de l'entraînement SFT.

Pourquoi : avec 36 questions, une question vaut 2,8 points et un écart de moins de 3 questions est du bruit. Ici ~190
questions (1 question ≈ 0,5 point) avec intervalle de confiance de Wilson à 95 %.

Garde-fous d'honnêteté :
  * aucune de ces questions n'est produite par les générateurs du SFT (synthetic_qa, synthetic_plus, personnalité) :
    `leakage()` le vérifie et un test échoue sinon ;
  * les pays/notions testés sont DIFFÉRENTS de ceux que le SFT enseigne (ex. le SFT enseigne 70 capitales ; ici 48 autres)
    -> on mesure ce que le modèle sait du pré-entraînement, pas ce qu'il a récité ;
  * ne couvre pas les sources externes du SFT (French Alpaca, PIAF, OASST) : un recoupement exact y est improbable mais non vérifié.

Notation : un mot attendu doit figurer comme MOT entier dans la réponse (accents/casse ignorés) — voir basics_eval.is_correct.
Les réponses attendues sont courtes et univoques ; les cas discutables (ex. capitale de la Tanzanie) sont volontairement absents.

Usage : python knowledge_eval.py --ckpt checkpoints/sft_C/final.pt
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from basics_eval import is_correct

Item = Tuple[str, str, List[str]]          # (catégorie, question, réponses acceptées)


def _de(word: str) -> str:
    """« de large » mais « d'ancien » / « d'oncle » : élision devant une voyelle (les mots à article gardent « de la … »)."""
    return ("d'" if word[:1].lower() in "aeiouyéèêœ" else "de ") + word


def _caps() -> List[Item]:
    # (« de + pays » avec son article, capitale acceptée(s))
    data = [
        # Afrique
        ("du Soudan", ["khartoum"]), ("de l'Ouganda", ["kampala"]), ("du Zimbabwe", ["harare"]), ("de la Namibie", ["windhoek"]),
        ("du Botswana", ["gaborone"]), ("du Mozambique", ["maputo"]), ("de la Somalie", ["mogadiscio", "mogadishu"]),
        ("de l'Érythrée", ["asmara"]), ("du Libéria", ["monrovia"]), ("de la Sierra Leone", ["freetown"]),
        ("de la Gambie", ["banjul"]), ("de la Guinée-Bissau", ["bissau"]), ("du Malawi", ["lilongwe"]),
        ("du Lesotho", ["maseru"]), ("de la Guinée équatoriale", ["malabo"]), ("du Soudan du Sud", ["djouba", "juba"]),
        ("des Comores", ["moroni"]), ("de l'île Maurice", ["port-louis", "port louis"]), ("du Cap-Vert", ["praia"]),
        ("de Djibouti", ["djibouti"]),
        # Europe
        ("de l'Islande", ["reykjavik"]), ("de la Croatie", ["zagreb"]), ("de la Serbie", ["belgrade"]),
        ("de la Bulgarie", ["sofia"]), ("de la Slovaquie", ["bratislava"]), ("de la Slovénie", ["ljubljana"]),
        ("de la République tchèque", ["prague"]), ("de la Lituanie", ["vilnius"]), ("de la Lettonie", ["riga"]),
        ("de l'Estonie", ["tallinn"]), ("de la Biélorussie", ["minsk"]), ("de l'Albanie", ["tirana"]),
        # Asie
        ("du Pakistan", ["islamabad"]), ("du Bangladesh", ["dacca", "dhaka"]), ("du Népal", ["katmandou", "kathmandou", "kathmandu"]),
        ("du Cambodge", ["phnom penh"]), ("du Laos", ["vientiane"]), ("de la Malaisie", ["kuala lumpur"]),
        ("des Philippines", ["manille"]), ("de la Mongolie", ["oulan-bator", "oulan bator", "ulaanbaatar"]),
        ("de la Corée du Nord", ["pyongyang"]), ("de la Syrie", ["damas"]),
        # Amériques, Océanie
        ("de l'Équateur", ["quito"]), ("du Venezuela", ["caracas"]), ("du Paraguay", ["asuncion"]),
        ("de l'Uruguay", ["montevideo"]), ("d'Haïti", ["port-au-prince", "port au prince"]),
        ("de la Nouvelle-Zélande", ["wellington"]),
    ]
    return [("capitales", f"Quelle est la capitale {de} ?", ans) for de, ans in data]


def _regions() -> List[Item]:
    data = [("de l'Adamaoua", "ngaoundere"), ("du Centre", "yaounde"), ("de l'Est", "bertoua"), ("de l'Extrême-Nord", "maroua"),
            ("du Littoral", "douala"), ("du Nord", "garoua"), ("du Nord-Ouest", "bamenda"), ("de l'Ouest", "bafoussam"),
            ("du Sud", "ebolowa"), ("du Sud-Ouest", "buea")]
    out: List[Item] = [("cameroun", f"Quel est le chef-lieu de la région {de} au Cameroun ?", [a]) for de, a in data]
    out += [
        ("cameroun", "Quel fleuve traverse la ville de Garoua ?", ["benoue"]),
        ("cameroun", "Quel est le plus haut sommet du Cameroun ?", ["mont cameroun"]),
        ("cameroun", "Qui a été le premier président du Cameroun ?", ["ahidjo"]),
        ("cameroun", "Quel lac est partagé entre le Cameroun, le Niger, le Nigeria et un autre pays voisin ?", ["lac tchad", "tchad"]),
        ("cameroun", "Quel animal symbolise l'équipe nationale de football du Cameroun ?", ["lion", "lions"]),
        ("cameroun", "Quelle ville du Cameroun est surnommée la ville aux sept collines ?", ["yaounde"]),
    ]
    return out


GEO: List[Item] = [
    ("géographie", "Quel est le plus grand pays du monde par la superficie ?", ["russie"]),
    ("géographie", "Quel est le plus petit pays du monde ?", ["vatican"]),
    ("géographie", "Quel est le plus haut sommet d'Afrique ?", ["kilimandjaro", "kilimanjaro"]),
    ("géographie", "Quel océan borde l'Afrique à l'ouest ?", ["atlantique"]),
    ("géographie", "Quelle mer sépare l'Europe de l'Afrique ?", ["mediterranee"]),
    ("géographie", "Quel fleuve traverse Paris ?", ["seine"]),
    ("géographie", "Quel fleuve traverse Londres ?", ["tamise"]),
    ("géographie", "Quel fleuve traverse Rome ?", ["tibre"]),
    ("géographie", "Sur quel continent se trouve l'Égypte ?", ["afrique"]),
    ("géographie", "Sur quel continent se trouve le Brésil ?", ["amerique du sud", "amerique latine"]),
    ("géographie", "Sur quel continent se trouve le Japon ?", ["asie"]),
    ("géographie", "Sur quel continent se trouve l'Allemagne ?", ["europe"]),
    ("géographie", "Dans quel pays se trouvent les pyramides de Gizeh ?", ["egypte"]),
    ("géographie", "Dans quel pays se trouve le Taj Mahal ?", ["inde"]),
    ("géographie", "Dans quel pays se trouve la Grande Muraille ?", ["chine"]),
    ("géographie", "Quelle est la plus grande île du monde ?", ["groenland"]),
    ("géographie", "Quel pays a la forme d'une botte ?", ["italie"]),
    ("géographie", "Quel est le plus grand lac d'Afrique ?", ["victoria"]),
]

SCIENCE: List[Item] = [
    ("science", "Quel métal précieux a pour symbole chimique Au ?", ["or"]),
    ("science", "Quel métal a pour symbole chimique Fe ?", ["fer"]),
    ("science", "Quel élément a pour symbole chimique Na ?", ["sodium"]),
    ("science", "Quel métal précieux a pour symbole chimique Ag ?", ["argent"]),
    ("science", "Quelle est la formule chimique du dioxyde de carbone ?", ["co2"]),
    ("science", "Quelle est la formule chimique du sel de cuisine ?", ["nacl"]),
    ("science", "Quel gaz les plantes absorbent-elles pour faire la photosynthèse ?", ["dioxyde de carbone", "co2", "gaz carbonique"]),
    ("science", "Quel gaz respirons-nous pour vivre ?", ["oxygene"]),
    ("science", "Quelle planète est la plus éloignée du Soleil ?", ["neptune"]),
    ("science", "Quelle planète est entourée d'anneaux célèbres ?", ["saturne"]),
    ("science", "Quel est le satellite naturel de la Terre ?", ["lune"]),
    ("science", "Quelle étoile se trouve au centre du système solaire ?", ["soleil"]),
    ("science", "Quelle force nous attire vers le sol ?", ["gravite", "gravitation", "pesanteur"]),
    ("science", "Quel organe pompe le sang dans le corps ?", ["coeur", "cœur"]),
    ("science", "Avec quel organe respire-t-on ?", ["poumon", "poumons"]),
    ("science", "Quel métal est liquide à température ambiante ?", ["mercure"]),
    ("science", "Quelle est la planète la plus chaude du système solaire ?", ["venus"]),
    ("science", "Quelle planète est connue pour sa Grande Tache Rouge ?", ["jupiter"]),
]

ANIMALS: List[Item] = [
    ("animaux", "Comment appelle-t-on le petit du cheval ?", ["poulain"]),
    ("animaux", "Comment appelle-t-on le petit de la chèvre ?", ["chevreau"]),
    ("animaux", "Comment appelle-t-on le petit du mouton ?", ["agneau"]),
    ("animaux", "Comment appelle-t-on le petit de la poule ?", ["poussin"]),
    ("animaux", "Comment appelle-t-on le petit du lion ?", ["lionceau"]),
    ("animaux", "Comment appelle-t-on le petit de l'ours ?", ["ourson"]),
    ("animaux", "Comment appelle-t-on le petit du canard ?", ["caneton"]),
    ("animaux", "Comment appelle-t-on le petit du cerf ?", ["faon"]),
    ("animaux", "Quel est le plus grand animal terrestre ?", ["elephant"]),
    ("animaux", "Quel est le plus grand animal du monde ?", ["baleine bleue", "baleine"]),
    ("animaux", "Quel est l'animal terrestre le plus rapide ?", ["guepard"]),
    ("animaux", "Quel animal fait cocorico ?", ["coq"]),
    ("animaux", "Quel animal produit la laine ?", ["mouton", "brebis"]),
    ("animaux", "Combien de pattes a un insecte ?", ["six", "6"]),
    ("animaux", "Combien de pattes a un chien ?", ["quatre", "4"]),
]

NUMBERS: List[Item] = [
    ("nombres", "Combien de côtés a un hexagone ?", ["six", "6"]),
    ("nombres", "Combien de côtés a un pentagone ?", ["cinq", "5"]),
    ("nombres", "Combien de côtés a un rectangle ?", ["quatre", "4"]),
    ("nombres", "Combien de côtés a un octogone ?", ["huit", "8"]),
    ("nombres", "Combien de degrés mesure un angle droit ?", ["90", "quatre-vingt-dix"]),
    ("nombres", "Quelle est la somme des angles d'un triangle, en degrés ?", ["180", "cent quatre-vingts"]),
    ("nombres", "Combien de centimètres y a-t-il dans un mètre ?", ["100", "cent"]),
    ("nombres", "Combien de mètres y a-t-il dans un kilomètre ?", ["1000", "mille"]),
    ("nombres", "Combien de grammes y a-t-il dans un kilogramme ?", ["1000", "mille"]),
    ("nombres", "Combien de semaines y a-t-il dans une année ?", ["52", "cinquante-deux"]),
    ("nombres", "Combien de jours compte le mois de février dans une année normale ?", ["28", "vingt-huit"]),
    ("nombres", "Combien de jours compte le mois de juin ?", ["30", "trente"]),
    ("nombres", "Combien de minutes y a-t-il dans une demi-heure ?", ["30", "trente"]),
    ("nombres", "Combien de joueurs une équipe de football aligne-t-elle sur le terrain ?", ["11", "onze"]),
]

LANGUAGE: List[Item] = (
    [("langue", f"Quel est le féminin {_de(m)} ?", [f]) for m, f in
     [("beau", "belle"), ("nouveau", "nouvelle"), ("vieux", "vieille"), ("blanc", "blanche"), ("heureux", "heureuse"),
      ("long", "longue"), ("doux", "douce"), ("gentil", "gentille"), ("frère", "soeur"), ("père", "mere"),
      ("oncle", "tante"), ("neveu", "niece"), ("prince", "princesse"), ("coq", "poule"), ("dieu", "deesse"), ("cheval", "jument")]]
    + [("langue", f"Quel est le pluriel {_de(s)} ?", [p]) for s, p in
       [("cheval", "chevaux"), ("journal", "journaux"), ("genou", "genoux"), ("chou", "choux"), ("bocal", "bocaux"),
        ("canal", "canaux"), ("œil", "yeux"), ("monsieur", "messieurs")]]
    + [("langue", f"Quel est le contraire {_de(w)} ?", a) for w, a in
       [("large", ["etroit"]), ("ancien", ["nouveau", "moderne", "recent", "neuf"]), ("la paix", ["guerre"]),
        ("accepter", ["refuser"]), ("pousser", ["tirer"]), ("la gauche", ["droite"]), ("le nord", ["sud"]),
        ("la naissance", ["mort", "deces"]), ("le silence", ["bruit"]), ("devant", ["derriere"])]]
    + [("langue", f"Quel est un synonyme {_de(w)} ?", a) for w, a in
       [("gentil", ["aimable", "bon", "sympathique", "agreable"]), ("triste", ["malheureux", "chagrine", "melancolique", "abattu", "peine"]),
        ("petit", ["minuscule", "menu", "mini", "reduit"]), ("chemin", ["route", "sentier", "voie", "piste"]),
        ("regarder", ["observer", "contempler", "fixer", "voir"])]]
    + [("langue", f"Quel est le participe passé du verbe {v} ?", [p]) for v, p in
       [("manger", "mange"), ("finir", "fini"), ("prendre", "pris"), ("voir", "vu"), ("avoir", "eu")]]
    + [("langue", "Quel est l'infinitif du verbe dans la phrase « je vais » ?", ["aller"])]
)

CULTURE: List[Item] = [
    ("culture", "Qui a écrit Le Petit Prince ?", ["saint-exupery", "exupery"]),
    ("culture", "Qui a écrit Roméo et Juliette ?", ["shakespeare"]),
    ("culture", "Qui a composé la Neuvième Symphonie avec l'Ode à la joie ?", ["beethoven"]),
    ("culture", "Qui a peint La Nuit étoilée ?", ["van gogh"]),
    ("culture", "Qui a peint Guernica ?", ["picasso"]),
    ("culture", "Qui a écrit la fable Le Corbeau et le Renard ?", ["la fontaine"]),
    ("culture", "Qui a développé la théorie de la relativité ?", ["einstein"]),
    ("culture", "Qui a été le premier homme à marcher sur la Lune ?", ["armstrong"]),
    ("culture", "Quel pays a offert la statue de la Liberté aux États-Unis ?", ["france"]),
    ("culture", "Dans quelle ville se trouve le Colisée ?", ["rome"]),
    ("culture", "Dans quelle ville se trouve Big Ben ?", ["londres"]),
    ("culture", "Qui est considéré comme l'inventeur du téléphone ?", ["bell"]),
    ("culture", "Qui a traversé l'Atlantique en 1492 et atteint l'Amérique ?", ["colomb"]),
    ("culture", "Qui a écrit Le Rouge et le Noir ?", ["stendhal"]),
    ("culture", "Qui a écrit Germinal ?", ["zola"]),
    ("culture", "Qui a été le premier président des États-Unis ?", ["washington"]),
    ("culture", "En quelle année a commencé la Révolution française ?", ["1789"]),
    ("culture", "Quelle est la langue officielle du Brésil ?", ["portugais"]),
    ("culture", "Quelle langue parle-t-on principalement en Espagne ?", ["espagnol", "castillan"]),
    ("culture", "Quelle est la monnaie du Japon ?", ["yen"]),
    ("culture", "Quelle est la monnaie des États-Unis ?", ["dollar", "dollars"]),
    ("culture", "Quelle est la monnaie du Nigeria ?", ["naira"]),
]

TECH: List[Item] = [
    ("informatique", "Que signifie l'acronyme IA ?", ["intelligence artificielle"]),
    ("informatique", "Que signifie l'acronyme LLM ?", ["grand modele de langage", "large language model", "modele de langage"]),
    ("informatique", "Que signifie l'acronyme RAM ?", ["memoire vive", "random access memory"]),
    ("informatique", "Quel langage de programmation porte le nom d'un serpent ?", ["python"]),
    ("informatique", "Quel système d'exploitation a pour mascotte un manchot ?", ["linux"]),
    ("informatique", "Combien de bits y a-t-il dans un octet ?", ["8", "huit"]),
    ("informatique", "Quelle entreprise a créé le système Windows ?", ["microsoft"]),
    ("informatique", "Que signifie l'acronyme HTML ?", ["hypertext", "hypertexte"]),
]

KNOWLEDGE: List[Item] = _caps() + _regions() + GEO + SCIENCE + ANIMALS + NUMBERS + LANGUAGE + CULTURE + TECH


# ── intervalle de confiance de Wilson (95 %) ───────────────────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(ids_a: List[int], ids_b: List[int]) -> Tuple[int, int, float]:
    """Comparaison APPARIÉE de deux modèles sur les mêmes questions (indices des questions réussies).
    Renvoie (a seul juste, b seul juste, p-valeur bilatérale du test exact de McNemar). Plus fin que de comparer deux
    intervalles de confiance : seules comptent les questions où les deux modèles diffèrent."""
    sa, sb = set(ids_a), set(ids_b)
    only_a, only_b = len(sa - sb), len(sb - sa)
    n = only_a + only_b
    if n == 0:
        return 0, 0, 1.0
    kmin = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(kmin + 1)) / 2 ** n
    return only_a, only_b, min(1.0, 2 * tail)


def leakage() -> List[str]:
    """Questions de ce test qui sortent d'un générateur du SFT (synthétique ancien, synthetic_plus plein et allégé, personnalité)."""
    import os

    from synthetic_plus import build_synthetic_plus, norm_q
    from synthetic_qa import build_synthetic_qa

    train = set()
    for r in range(6):
        for c in build_synthetic_qa(seed=r):
            train |= {norm_q(m["content"]) for m in c["messages"] if m["role"] == "user"}
    for light in (False, True):
        for c in build_synthetic_plus(seed=0, light=light):
            train |= {norm_q(m["content"]) for m in c["messages"] if m["role"] == "user"}
    here = os.path.dirname(os.path.abspath(__file__))
    pj = os.path.join(here, "personnalite.jsonl")
    if os.path.exists(pj):
        for line in open(pj, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                msgs = d.get("messages") or [{"role": "user", "content": d.get(k, "")} for k in ("question", "instruction", "prompt") if k in d]
                train |= {norm_q(m["content"]) for m in msgs if m.get("role") == "user"}
    return [q for _, q, _ in KNOWLEDGE if norm_q(q) in train]


def summarize(results: Dict[str, List[bool]]) -> Dict[str, object]:
    allk = sum(sum(v) for v in results.values())
    alln = sum(len(v) for v in results.values())
    lo, hi = wilson(allk, alln)
    return {"n": alln, "correct": allk, "knowledge": round(allk / alln, 3) if alln else 0.0,
            "ci95": [round(lo, 3), round(hi, 3)],
            "par_categorie": {c: {"n": len(v), "acc": round(sum(v) / len(v), 3)} for c, v in sorted(results.items())}}


def main():
    p = argparse.ArgumentParser(description="MiniLLM — test de connaissances tenues hors du SFT")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    from evaluate import chat_reply, load_model     # même API que skills_eval.py / basics_eval.py (validée sur Colab)

    model, tok, _ = load_model(a.ckpt, device=a.device)
    results: Dict[str, List[bool]] = defaultdict(list)
    ok_ids: List[int] = []
    shown = 0
    for idx, (cat, q, accepted) in enumerate(KNOWLEDGE):
        reply = chat_reply(model, tok, [{"role": "user", "content": q}], max_new_tokens=60,
                           temperature=0.0, repetition_penalty=1.0)
        ok = is_correct(reply, accepted)
        results[cat].append(ok)
        if ok:
            ok_ids.append(idx)
        if not ok and shown < 6:
            print(f"  ✗ [{cat}] {q}\n      attendu : {accepted[0]} | obtenu : {reply[:90]}")
            shown += 1
    res = summarize(results)
    res["ok_ids"] = ok_ids                                  # pour la comparaison appariée (mcnemar_exact)
    for cat, d in res["par_categorie"].items():
        print(f"  {cat:<14} {d['acc'] * 100:5.1f} %  (n={d['n']})")
    print(f"knowledge : {res['correct']}/{res['n']} = {res['knowledge'] * 100:.1f} %  IC95 [{res['ci95'][0] * 100:.1f} ; {res['ci95'][1] * 100:.1f}]")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
