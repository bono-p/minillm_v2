"""
synthetic_plus.py — données synthétiques supplémentaires pour le SFT v3.1 (100 % hors-ligne, calculées par code).

Pourquoi : dans le SFT v3, le synthétique ne comptait que ~3 150 exemples uniques (dont ~1 000 d'arithmétique) ;
le modèle se trompait encore sur 7+8 ou 12×12. Ici on couvre PLUS de cas, avec le calcul écrit AVANT le résultat
dans les problèmes ("10 - 3 = 7. Il te reste 7 pommes."), ce qui est plus facile à apprendre pour un petit modèle.

Deux garde-fous d'honnêteté :
  * `exclude` : toute question déjà présente dans la batterie de test (UNSEEN_QA_PROMPTS) est retirée de l'entraînement ;
  * `is_heldout(key)` : 10 % des problèmes chiffrés ne sont JAMAIS générés pour l'entraînement ; `skills_eval.py` ne
    teste que ceux-là -> on mesure une vraie généralisation, pas de la récitation.
"""
from __future__ import annotations

import random
import re
import zlib
from typing import Dict, Iterable, List, Optional, Set

from synthetic_qa import DAYS, _conv


def is_heldout(key: str) -> bool:
    """Vrai pour ~10 % des problèmes : réservés à l'évaluation des compétences, absents de l'entraînement."""
    return zlib.crc32(key.encode("utf-8")) % 10 == 0


def norm_q(q: str) -> str:
    # on GARDE les symboles (+ - × ÷) : « 23 + 17 » et « 23 - 17 » sont deux questions différentes
    return re.sub(r"\s+", " ", q.lower().replace("’", "'")).strip(" ?!.")


# ── arithmétique ───────────────────────────────────────────────────────────────
def _arith_item(op: str, a: int, b: int):
    if op == "add":
        return a + b, "plus", "+"
    if op == "sub":
        return a - b, "moins", "-"
    if op == "mul":
        return a * b, "fois", "×"
    return a // b, "divisé par", "÷"


def arithmetic_pairs(op: str):
    """Tous les couples (a, b) d'une opération (réponses entières, positives)."""
    if op == "add":
        return [(a, b) for a in range(0, 101) for b in range(0, 101 - 0)]
    if op == "sub":
        return [(a, b) for a in range(0, 101) for b in range(0, a + 1)]
    if op == "mul":
        return [(a, b) for a in range(0, 13) for b in range(0, 13)]
    return [(a * b, b) for a in range(0, 13) for b in range(1, 13)]


def arithmetic_question(op: str, a: int, b: int, rng: Optional[random.Random] = None) -> Dict:
    r, word, sym = _arith_item(op, a, b)
    rng = rng or random.Random(0)
    if rng.random() < 0.65:
        q = rng.choice([f"Combien font {a} {word} {b} ?", f"Combien fait {a} {word} {b} ?", f"{a} {word} {b} ?"])
        ans = f"{a} {word} {b} font {r}." if r != 1 else f"{a} {word} {b} fait 1."
    else:
        q = f"Combien font {a} {sym} {b} ?"
        ans = f"{a} {sym} {b} = {r}."
    return _conv(q, ans, "arithmetic_plus")


def _arithmetic(rng: random.Random, n_add=2600, n_sub=1800, n_mul=169, n_div=144) -> List[Dict]:
    out: List[Dict] = []
    for op, n in (("add", n_add), ("sub", n_sub), ("mul", n_mul), ("div", n_div)):
        pairs = [p for p in arithmetic_pairs(op) if not is_heldout(f"{op}|{p[0]}|{p[1]}")]
        rng.shuffle(pairs)
        for a, b in pairs[:n]:
            out.append(arithmetic_question(op, a, b, rng))
    # pourcentages simples (réponse entière)
    for pct in (10, 25, 50):
        for base in range(20, 401, 20):
            if (base * pct) % 100:
                continue
            key = f"pct|{pct}|{base}"
            if is_heldout(key):
                continue
            out.append(_conv(f"Combien font {pct} pour cent de {base} ?",
                             f"{pct} pour cent de {base} = {base * pct // 100}.", "arithmetic_plus"))
    # carrés (le carré de n : n × n)
    for n in range(2, 16):
        if not is_heldout(f"sq|{n}"):
            out.append(_conv(f"Quel est le carré de {n} ?", f"Le carré de {n} est {n} × {n} = {n * n}.", "arithmetic_plus"))
    return out


# ── petits problèmes : le calcul est écrit avant le résultat ──────────────────────
OBJECTS = ["pommes", "billes", "bonbons", "livres", "stylos", "mangues", "cahiers", "oranges", "crayons", "œufs"]
NAMES = ["Paul", "Jean", "Marc", "Awa", "Fatou", "Ali", "Sophie", "Moussa", "Clara", "Yannick", "Aïcha", "Luc"]


def problem_keys_and_text(kind: str, rng: random.Random):
    """Renvoie (key, question, réponse, résultat) pour un problème de type `kind`."""
    if kind == "give":
        a, o = rng.randint(5, 60), rng.choice(OBJECTS)
        b = rng.randint(1, a - 1)
        r = a - b
        return (f"give|{a}|{b}", f"Si j'ai {a} {o} et que j'en donne {b}, combien m'en reste-t-il ?",
                f"{a} - {b} = {r}. Il te reste {r} {o}.", r)
    if kind == "price":
        p, n = rng.choice([500, 1000, 1500, 2000, 2500, 3000]), rng.randint(2, 9)
        return (f"price|{p}|{n}", f"Si un livre coûte {p} francs et que j'en achète {n}, combien dois-je payer ?",
                f"{n} × {p} = {n * p}. Tu dois payer {n * p} francs.", n * p)
    if kind == "speed":
        v, h = rng.choice([40, 50, 60, 80, 90, 100]), rng.randint(2, 6)
        return (f"speed|{v}|{h}", f"Une voiture roule à {v} km par heure. Quelle distance parcourt-elle en {h} heures ?",
                f"{v} × {h} = {v * h}. Elle parcourt {v * h} km.", v * h)
    if kind == "share":
        n, each = rng.randint(2, 9), rng.randint(2, 12)
        return (f"share|{n}|{each}", f"{n} personnes se partagent équitablement {n * each} bonbons. Combien chaque personne reçoit-elle ?",
                f"{n * each} ÷ {n} = {each}. Chaque personne reçoit {each} bonbons.", each)
    if kind == "class":
        t, f = rng.randint(20, 45), rng.randint(5, 19)
        return (f"class|{t}|{f}", f"Une classe contient {t} élèves. {f} sont des filles. Combien y a-t-il de garçons ?",
                f"{t} - {f} = {t - f}. Il y a {t - f} garçons.", t - f)
    if kind == "train":
        h1 = rng.randint(5, 15)
        d = rng.randint(1, 8)
        return (f"train|{h1}|{d}", f"Un train part à {h1} heures et arrive à {h1 + d} heures. Combien de temps dure le trajet ?",
                f"{h1 + d} - {h1} = {d}. Le trajet dure {d} heures.", d)
    if kind == "money":
        m, d = rng.choice([2000, 3000, 5000, 10000]), rng.choice([250, 500, 750, 1000, 1250, 1750])
        return (f"money|{m}|{d}", f"J'ai {m} francs et je dépense {d} francs. Combien me reste-t-il ?",
                f"{m} - {d} = {m - d}. Il te reste {m - d} francs.", m - d)
    if kind == "pens":
        n, c, k = rng.randint(2, 6), rng.choice([250, 500, 750]), rng.randint(2, 4)
        return (f"pens|{n}|{c}|{k}", f"Si {n} stylos coûtent {n * c} francs, combien coûtent {n * k} stylos ?",
                f"Un stylo coûte {n * c} ÷ {n} = {c} francs. Donc {n * k} stylos coûtent {n * k} × {c} = {n * k * c} francs.", n * k * c)
    # bottles
    l, n = rng.choice([1, 2, 3, 5]), rng.randint(2, 8)
    return (f"bottles|{l}|{n}", f"Une bouteille contient {l} litres. Combien contiennent {n} bouteilles ?",
            f"{n} × {l} = {n * l}. {n} bouteilles contiennent {n * l} litres.", n * l)


PROBLEM_KINDS = ["give", "price", "speed", "share", "class", "train", "money", "pens", "bottles"]


def _problems(rng: random.Random, per_kind: int = 220) -> List[Dict]:
    out, seen = [], set()
    for kind in PROBLEM_KINDS:
        tries = 0
        got = 0
        while got < per_kind and tries < per_kind * 8:
            tries += 1
            key, q, a, _ = problem_keys_and_text(kind, rng)
            if key in seen or is_heldout(key):
                continue
            seen.add(key)
            out.append(_conv(q, a, "problems"))
            got += 1
    return out


# ── suites de nombres ────────────────────────────────────────────────────────────
def _sequences(rng: random.Random) -> List[Dict]:
    out = []
    for step in range(1, 11):
        for start in range(0, 31):
            for length in (4, 5):
                seq = [start + step * i for i in range(length)]
                key = f"seq|{start}|{step}|{length}"
                if is_heldout(key):
                    continue
                nxt = start + step * length
                shown = ", ".join(map(str, seq))
                out.append(_conv(f"Quel nombre vient après {shown} ?", f"Après {shown}, c'est {nxt} : on ajoute {step} à chaque fois.", "sequences"))
                if length == 5:
                    k = rng.randint(1, 3)
                    gap = ", ".join("?" if i == k else str(v) for i, v in enumerate(seq))
                    out.append(_conv(f"Quel nombre manque : {gap} ?", f"Le nombre qui manque est {seq[k]}.", "sequences"))
    rng.shuffle(out)
    return out[:1500]


# ── intrus, logique ──────────────────────────────────────────────────────────────
GROUPS = [
    ("un fruit", ["pomme", "banane", "orange", "mangue", "papaye", "ananas", "goyave", "citron"]),
    ("un animal", ["chien", "chat", "cheval", "lion", "vache", "chèvre", "éléphant", "poule"]),
    ("un légume", ["carotte", "tomate", "oignon", "poireau", "haricot", "courgette", "aubergine"]),
    ("un véhicule", ["voiture", "moto", "bus", "camion", "vélo", "train", "bateau"]),
    ("une couleur", ["rouge", "bleu", "vert", "jaune", "noir", "blanc", "violet"]),
    ("un vêtement", ["pantalon", "chemise", "robe", "chapeau", "jupe", "veste", "écharpe"]),
    ("un outil", ["marteau", "scie", "tournevis", "pince", "pelle", "râteau"]),
    ("un instrument de musique", ["guitare", "piano", "violon", "flûte", "tambour", "trompette"]),
]


def _odd_one_out(rng: random.Random) -> List[Dict]:
    out = []
    for _ in range(500):
        (art, items), (_, others) = rng.sample(GROUPS, 2)
        three = rng.sample(items, 3)
        odd = rng.choice(others)
        words = three + [odd]
        rng.shuffle(words)
        shown = ", ".join(words[:-1]) + " et " + words[-1]
        out.append(_conv(f"Quel est l'intrus entre {shown} ?", f"L'intrus est {odd}, car ce n'est pas {art}.", "logic"))
    return out


ADJ = [  # (adjectif, comparatif, le plus grand, le plus petit)
    ("âgé", "plus âgé", "le plus âgé", "le plus jeune"),
    ("grand", "plus grand", "le plus grand", "le plus petit"),
    ("rapide", "plus rapide", "le plus rapide", "le plus lent"),
    ("lourd", "plus lourd", "le plus lourd", "le plus léger"),
    ("fort", "plus fort", "le plus fort", "le plus faible"),
]
SYLLOGISM = [  # (pluriel x, pluriel y, "un x", "un y")
    ("chats", "animaux", "un chat", "un animal"), ("roses", "fleurs", "une rose", "une fleur"),
    ("mangues", "fruits", "une mangue", "un fruit"), ("carrés", "rectangles", "un carré", "un rectangle"),
    ("chiens", "mammifères", "un chien", "un mammifère"), ("pommiers", "arbres", "un pommier", "un arbre"),
    ("lions", "félins", "un lion", "un félin"), ("moineaux", "oiseaux", "un moineau", "un oiseau"),
]


def _logic(rng: random.Random) -> List[Dict]:
    out = []
    for _ in range(400):
        a, b, c = rng.sample(NAMES, 3)
        adj, comp, big, small = rng.choice(ADJ)
        if rng.random() < 0.5:
            q = f"{a} est {comp} que {b}. {b} est {comp} que {c}. Qui est {small} ?"
            out.append(_conv(q, f"{c} est {small}.", "logic"))
        else:
            q = f"{a} est {comp} que {b}. {b} est {comp} que {c}. Qui est {big} ?"
            out.append(_conv(q, f"{a} est {big}.", "logic"))
    for px, py, sx, sy in SYLLOGISM:
        for name in NAMES[:6]:
            out.append(_conv(f"Si tous les {px} sont des {py} et que {name} est {sx}, que peut-on dire de {name} ?",
                             f"{name} est aussi {sy}.", "logic"))
    return out


# ── calendrier ───────────────────────────────────────────────────────────────────
def _calendar_plus(rng: random.Random) -> List[Dict]:
    out = []
    for i, d in enumerate(DAYS):
        out.append(_conv(f"Si aujourd'hui est {d}, quel jour sera demain ?", f"Si aujourd'hui est {d}, demain sera {DAYS[(i + 1) % 7]}.", "calendar_plus"))
        out.append(_conv(f"Si aujourd'hui est {d}, quel jour sera après-demain ?", f"Si aujourd'hui est {d}, après-demain sera {DAYS[(i + 2) % 7]}.", "calendar_plus"))
        out.append(_conv(f"Si aujourd'hui est {d}, quel jour était hier ?", f"Si aujourd'hui est {d}, hier c'était {DAYS[(i - 1) % 7]}.", "calendar_plus"))
        for k in (3, 4, 5, 6):
            out.append(_conv(f"Quel jour sera-t-il dans {k} jours si nous sommes {d} ?", f"Dans {k} jours, nous serons {DAYS[(i + k) % 7]}.", "calendar_plus"))
    return out


# ── grammaire / vocabulaire ───────────────────────────────────────────────────────
PLURALS = [("animal", "animaux"), ("oiseau", "oiseaux"), ("chapeau", "chapeaux"), ("travail", "travaux"),
           ("hôpital", "hôpitaux"), ("jeu", "jeux"), ("bijou", "bijoux"), ("avion", "avions"), ("enfant", "enfants"),
           ("château", "châteaux"), ("cheveu", "cheveux"), ("nez", "nez"), ("souris", "souris"), ("bateau", "bateaux"),
           ("feu", "feux"), ("pneu", "pneus"), ("mur", "murs"), ("tableau", "tableaux"), ("gâteau", "gâteaux"), ("fleur", "fleurs")]
FEMININES = [("infirmier", "infirmière"), ("boulanger", "boulangère"), ("étudiant", "étudiante"), ("chanteur", "chanteuse"),
             ("roi", "reine"), ("lion", "lionne"), ("chat", "chatte"), ("ami", "amie"), ("voisin", "voisine"),
             ("cousin", "cousine"), ("serveur", "serveuse"), ("danseur", "danseuse")]
SYNONYMS = [("commencer", "débuter"), ("finir", "terminer"), ("voiture", "automobile"), ("livre", "ouvrage"), ("peur", "crainte"),
            ("fatigué", "épuisé"), ("parler", "discuter"), ("maison", "habitation"), ("travail", "emploi"), ("rapide", "véloce"),
            ("joli", "beau"), ("content", "heureux")]
MORE_OPPOSITES = [("monter", "descendre"), ("entrer", "sortir"), ("acheter", "vendre"), ("aimer", "détester"), ("gagner", "perdre"),
                  ("avant", "après"), ("jour", "nuit"), ("riche", "pauvre"), ("sec", "mouillé"), ("dur", "mou"),
                  ("long", "court"), ("épais", "mince"), ("beau", "laid"), ("tôt", "tard"), ("début", "fin"),
                  ("ami", "ennemi"), ("pair", "impair"), ("vrai", "faux"), ("dedans", "dehors"), ("allumer", "éteindre")]
CORRECTIONS = [
    ("Elle sont partie hier.", "Elles sont parties hier."), ("Il mange des pomme.", "Il mange des pommes."),
    ("Nous avons manger du riz.", "Nous avons mangé du riz."), ("Les fille chante.", "Les filles chantent."),
    ("Tu a raison.", "Tu as raison."), ("Je suis aller au marché.", "Je suis allé au marché."),
    ("Mon amie est content.", "Mon amie est contente."), ("Les chats noir dorment.", "Les chats noirs dorment."),
    ("Il fait beau, mais je reste a la maison.", "Il fait beau, mais je reste à la maison."),
    ("Ils va à l'école.", "Ils vont à l'école."), ("Vous avez fini vos devoir.", "Vous avez fini vos devoirs."),
    ("La maison est grande et beaux.", "La maison est grande et belle."), ("Je mange du pain et du fromages.", "Je mange du pain et du fromage."),
    ("Mes parents est arrivés.", "Mes parents sont arrivés."),
]
CORRECT_ALREADY = ["Le chat dort sur le canapé.", "Nous allons à l'école.", "Elle a mangé une mangue.", "Les enfants jouent dans la cour.",
                   "Mon frère travaille à Douala.", "Tu as raison."]


def _language(rng: random.Random) -> List[Dict]:
    out = []
    for w, p in PLURALS:
        out += [_conv(f"Quel est le pluriel de {w} ?", f"Le pluriel de {w} est {p}.", "language"),
                _conv(f"Donne le pluriel de {w}.", f"Le pluriel de {w} est {p}.", "language")]
    for m, f in FEMININES:
        out += [_conv(f"Quel est le féminin de {m} ?", f"Le féminin de {m} est {f}.", "language"),
                _conv(f"Quel est le masculin de {f} ?", f"Le masculin de {f} est {m}.", "language")]
    for a, b in SYNONYMS:
        out += [_conv(f"Quel est le synonyme de {a} ?", f"Un synonyme de {a} est {b}.", "language"),
                _conv(f"Donne un synonyme de {a}.", f"Un synonyme de {a} est {b}.", "language")]
    for a, b in MORE_OPPOSITES:
        out += [_conv(f"Quel est le contraire de {a} ?", f"Le contraire de {a} est {b}.", "language"),
                _conv(f"Quel est le contraire de {b} ?", f"Le contraire de {b} est {a}.", "language")]
    for wrong, right in CORRECTIONS:
        out += [_conv(f"Corrige cette phrase : {wrong}", right, "language"),
                _conv(f"Corrige la phrase suivante : {wrong}", right, "language")]
    for ok in CORRECT_ALREADY:
        out.append(_conv(f"Corrige cette phrase : {ok}", ok, "language"))
    return out


# ── définitions courtes (concepts ABSENTS de la batterie de test) ──────────────────────
DEFINITIONS = [
    ("un volcan", "Un volcan est une montagne par laquelle sortent de la lave, des cendres et des gaz venant de l'intérieur de la Terre."),
    ("un fleuve", "Un fleuve est un grand cours d'eau qui se jette dans la mer ou dans l'océan."),
    ("une île", "Une île est une terre entourée d'eau de tous les côtés."),
    ("une forêt", "Une forêt est un grand espace couvert d'arbres."),
    ("un lac", "Un lac est une grande étendue d'eau entourée de terre."),
    ("une plage", "Une plage est une bande de sable ou de galets au bord de la mer ou d'un lac."),
    ("le vent", "Le vent est de l'air en mouvement."),
    ("un nuage", "Un nuage est un ensemble de minuscules gouttes d'eau ou de cristaux de glace qui flottent dans le ciel."),
    ("un orage", "Un orage est un phénomène météorologique avec de la pluie, du tonnerre et des éclairs."),
    ("le climat", "Le climat est le temps qu'il fait en moyenne dans une région sur de nombreuses années."),
    ("un continent", "Un continent est une très grande étendue de terre entourée par les océans."),
    ("une capitale", "La capitale est la ville où se trouve le gouvernement d'un pays."),
    ("une frontière", "Une frontière est la limite qui sépare deux pays."),
    ("la population", "La population est l'ensemble des personnes qui vivent dans un lieu."),
    ("l'agriculture", "L'agriculture est l'activité qui consiste à cultiver la terre et à élever des animaux pour produire de la nourriture."),
    ("une monnaie", "Une monnaie est ce qu'on utilise pour acheter et vendre, comme les pièces et les billets."),
    ("une école", "Une école est un lieu où des élèves apprennent avec des enseignants."),
    ("un hôpital", "Un hôpital est un lieu où l'on soigne les malades et les blessés."),
    ("une bibliothèque", "Une bibliothèque est un lieu où l'on peut lire et emprunter des livres."),
    ("un musée", "Un musée est un lieu où l'on expose des objets d'art ou d'histoire."),
    ("un médicament", "Un médicament est un produit qui sert à soigner ou à prévenir une maladie."),
    ("une vitamine", "Une vitamine est une substance dont le corps a besoin en petite quantité pour bien fonctionner."),
    ("un squelette", "Le squelette est l'ensemble des os qui soutiennent le corps."),
    ("un muscle", "Un muscle est un organe qui se contracte pour produire un mouvement."),
    ("le cerveau", "Le cerveau est l'organe situé dans la tête qui commande le corps et permet de penser."),
    ("le sang", "Le sang est le liquide rouge qui circule dans le corps et transporte l'oxygène."),
    ("un oiseau", "Un oiseau est un animal couvert de plumes, qui a des ailes et pond des œufs."),
    ("un poisson", "Un poisson est un animal qui vit dans l'eau et respire grâce à des branchies."),
    ("un mammifère", "Un mammifère est un animal dont les petits se nourrissent du lait de leur mère."),
    ("un insecte", "Un insecte est un petit animal à six pattes, comme la fourmi ou l'abeille."),
    ("un arbre", "Un arbre est une grande plante avec un tronc en bois, des branches et des feuilles."),
    ("une graine", "Une graine est la partie d'une plante qui peut donner une nouvelle plante."),
    ("une racine", "La racine est la partie de la plante qui est dans le sol et qui absorbe l'eau."),
    ("un vaccin", "Un vaccin est un produit qui aide le corps à se protéger contre une maladie."),
    ("un thermomètre", "Un thermomètre est un instrument qui mesure la température."),
    ("une boussole", "Une boussole est un instrument dont l'aiguille indique le nord."),
    ("un miroir", "Un miroir est une surface qui renvoie l'image de ce qui se trouve devant."),
    ("un clavier", "Un clavier est un ensemble de touches qui sert à écrire sur un ordinateur."),
    ("un écran", "Un écran est la surface qui affiche les images d'un ordinateur ou d'un téléphone."),
    ("une imprimante", "Une imprimante est une machine qui imprime des documents sur du papier."),
    ("un mot de passe", "Un mot de passe est un code secret qui protège l'accès à un compte."),
    ("une application", "Une application est un programme qu'on utilise sur un téléphone ou un ordinateur."),
    ("un site web", "Un site web est un ensemble de pages accessibles sur Internet."),
    ("un logiciel", "Un logiciel est un ensemble de programmes qui permettent à un ordinateur d'effectuer des tâches."),
    ("un pixel", "Un pixel est le plus petit point d'une image affichée sur un écran."),
    ("un octet", "Un octet est une unité qui mesure la quantité d'informations stockées dans un ordinateur."),
    ("un virus informatique", "Un virus informatique est un programme malveillant qui peut endommager un ordinateur."),
    ("un séisme", "Un séisme, ou tremblement de terre, est un mouvement brusque du sol."),
    ("une rivière", "Une rivière est un cours d'eau qui se jette dans un fleuve ou dans un autre cours d'eau."),
    ("un village", "Un village est un petit groupe d'habitations, plus petit qu'une ville."),
    ("un marché", "Un marché est un lieu où l'on vend et où l'on achète des produits."),
    ("un pays", "Un pays est un territoire avec ses frontières, son gouvernement et sa population."),
    ("un engrais", "Un engrais est un produit qui apporte des nutriments aux plantes pour les aider à pousser."),
    ("la photosynthèse des plantes", "Les plantes utilisent la lumière du soleil, l'eau et le dioxyde de carbone pour fabriquer leur nourriture."),
    ("une saison", "Une saison est l'une des périodes de l'année qui se distinguent par le climat."),
    ("un siècle", "Un siècle est une période de cent ans."),
    ("une décennie", "Une décennie est une période de dix ans."),
    ("un kilomètre", "Un kilomètre est une unité de longueur qui vaut mille mètres."),
    ("un litre", "Un litre est une unité qui mesure le volume des liquides."),
    ("un gramme", "Un gramme est une unité de masse : mille grammes font un kilogramme."),
    ("un triangle", "Un triangle est une figure plane qui a trois côtés et trois angles."),
    ("un cercle", "Un cercle est une figure ronde dont tous les points sont à la même distance du centre."),
    ("un nombre pair", "Un nombre pair est un nombre divisible par deux, comme 2, 4 ou 10."),
    ("un nombre impair", "Un nombre impair est un nombre qui n'est pas divisible par deux, comme 3, 5 ou 11."),
    ("un nombre premier", "Un nombre premier est un nombre qui n'est divisible que par 1 et par lui-même, comme 2, 3, 5 ou 7."),
]


def _definitions(rng: random.Random) -> List[Dict]:
    out = []
    for phrase, definition in DEFINITIONS:
        if phrase.startswith(("un ", "une ")):
            q1 = f"Qu'est-ce qu'{phrase} ?"
        else:
            q1 = f"Qu'est-ce que {phrase} ?"
        out.append(_conv(q1, definition, "definitions"))
        out.append(_conv(f"Explique ce qu'est {phrase}.", definition, "definitions"))
    return out


# ── l'ANCIEN synthétique (synthetic_qa.py) contient déjà de l'arithmétique : on en retire les problèmes tenus à l'écart ──
_ARITH_RE = re.compile(r"(\d+) (plus|moins|fois|\+|-|×) (\d+)")
_OPKEY = {"plus": "add", "+": "add", "moins": "sub", "-": "sub", "fois": "mul", "×": "mul"}


def drop_heldout_arithmetic(convs: List[Dict]) -> List[Dict]:
    """Retire toute conversation dont une question utilisateur est un calcul réservé à l'évaluation (`is_heldout`)."""
    out = []
    for c in convs:
        held = False
        for m in c["messages"]:
            if m["role"] == "user":
                mt = _ARITH_RE.search(m["content"])
                if mt and is_heldout(f"{_OPKEY[mt.group(2)]}|{mt.group(1)}|{mt.group(3)}"):
                    held = True
        if not held:
            out.append(c)
    return out


PLUS_WEIGHTS = {"arithmetic_plus": 1.0, "problems": 1.0, "sequences": 1.0, "logic": 1.0, "calendar_plus": 2.0,
                "language": 1.5, "definitions": 3.0}


def build_synthetic_plus(seed: int = 0, exclude: Optional[Iterable[str]] = None, weights: Optional[Dict[str, float]] = None,
                         light: bool = False, arith_cap: int = 1000) -> List[Dict]:
    """Conversations synthétiques supplémentaires. `exclude` : questions à ne JAMAIS entraîner (batterie de test)."""
    rng = random.Random(seed)
    blocked: Set[str] = {norm_q(q) for q in (exclude or [])}
    w = {**PLUS_WEIGHTS, **(weights or {})}
    pools = {
        "arithmetic_plus": _arithmetic(rng), "problems": _problems(rng), "sequences": _sequences(rng),
        "logic": _odd_one_out(rng) + _logic(rng), "calendar_plus": _calendar_plus(rng),
        "language": _language(rng), "definitions": _definitions(rng),
    }
    if light:       # v3.2 : on garde ce qui a marché (logique, intrus, langue, calendrier) ; on retire ce qui n'a rien appris
        for useless in ("definitions", "sequences", "problems"):   # définitions : effondrement ; suites/problèmes : 0-5 % de réussite
            pools.pop(useless, None)
    out, seen = [], set()
    for cat, convs in pools.items():
        uniq = []
        for c in convs:
            q, a = c["messages"][0]["content"], c["messages"][1]["content"]
            if norm_q(q) in blocked or (q, a) in seen:
                continue
            seen.add((q, a))
            uniq.append(c)
        if light and cat == "arithmetic_plus":
            uniq = rng.sample(uniq, min(arith_cap, len(uniq)))
        n = round(len(uniq) * w.get(cat, 1.0))
        picked = [uniq[i % len(uniq)] for i in range(n)] if n > len(uniq) else rng.sample(uniq, n)
        out += picked
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    d = build_synthetic_plus()
    from collections import Counter
    print(len(d), "conversations", dict(Counter(c["category"] for c in d)))
    for c in d[:6]:
        print(c["messages"][0]["content"], "->", c["messages"][1]["content"])
