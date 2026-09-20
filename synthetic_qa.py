"""
synthetic_qa.py — jeu de questions/réponses SYNTHÉTIQUE, 100 % hors-ligne, généré par code.

But : apprendre à un tout petit modèle la FORME d'une conversation (question -> réponse courte, en phrase
complète, qui se termine) sur des faits simples et corrects (les réponses arithmétiques sont calculées).
Ce n'est pas de la "connaissance" : c'est ce qui permet au modèle de "faire semblant de répondre" proprement.

Contenu : identité, salutations, capitales, arithmétique, calendrier, faits simples, contraires,
questions auxquelles un petit modèle NE PEUT PAS répondre (météo, heure...) avec une réponse honnête,
et quelques conversations à 2 tours.
"""
from __future__ import annotations

import random
from typing import Dict, List

BOT_NAME = "MiniLLM"
BOT_MAKER = "DevLab"

# (nom avec article, "de" + nom, capitale)
CAPITALS = [
    ("la France", "de la France", "Paris"), ("le Cameroun", "du Cameroun", "Yaoundé"),
    ("le Nigeria", "du Nigeria", "Abuja"), ("le Tchad", "du Tchad", "N'Djamena"),
    ("le Sénégal", "du Sénégal", "Dakar"), ("la Côte d'Ivoire", "de la Côte d'Ivoire", "Yamoussoukro"),
    ("le Mali", "du Mali", "Bamako"), ("le Niger", "du Niger", "Niamey"),
    ("le Gabon", "du Gabon", "Libreville"), ("la République du Congo", "de la République du Congo", "Brazzaville"),
    ("la République démocratique du Congo", "de la République démocratique du Congo", "Kinshasa"),
    ("l'Algérie", "de l'Algérie", "Alger"), ("le Maroc", "du Maroc", "Rabat"),
    ("la Tunisie", "de la Tunisie", "Tunis"), ("l'Égypte", "de l'Égypte", "Le Caire"),
    ("l'Éthiopie", "de l'Éthiopie", "Addis-Abeba"), ("le Kenya", "du Kenya", "Nairobi"),
    ("le Ghana", "du Ghana", "Accra"), ("l'Italie", "de l'Italie", "Rome"),
    ("l'Espagne", "de l'Espagne", "Madrid"), ("l'Allemagne", "de l'Allemagne", "Berlin"),
    ("le Portugal", "du Portugal", "Lisbonne"), ("la Belgique", "de la Belgique", "Bruxelles"),
    ("la Suisse", "de la Suisse", "Berne"), ("le Royaume-Uni", "du Royaume-Uni", "Londres"),
    ("l'Irlande", "de l'Irlande", "Dublin"), ("les Pays-Bas", "des Pays-Bas", "Amsterdam"),
    ("la Grèce", "de la Grèce", "Athènes"), ("la Pologne", "de la Pologne", "Varsovie"),
    ("la Russie", "de la Russie", "Moscou"), ("la Turquie", "de la Turquie", "Ankara"),
    ("la Chine", "de la Chine", "Pékin"), ("le Japon", "du Japon", "Tokyo"),
    ("l'Inde", "de l'Inde", "New Delhi"), ("la Corée du Sud", "de la Corée du Sud", "Séoul"),
    ("le Canada", "du Canada", "Ottawa"), ("les États-Unis", "des États-Unis", "Washington"),
    ("le Mexique", "du Mexique", "Mexico"), ("le Brésil", "du Brésil", "Brasilia"),
    ("l'Argentine", "de l'Argentine", "Buenos Aires"), ("le Chili", "du Chili", "Santiago"),
    ("le Pérou", "du Pérou", "Lima"), ("la Colombie", "de la Colombie", "Bogota"),
    ("Cuba", "de Cuba", "La Havane"), ("l'Australie", "de l'Australie", "Canberra"),
    ("la Suède", "de la Suède", "Stockholm"), ("la Norvège", "de la Norvège", "Oslo"),
    ("le Danemark", "du Danemark", "Copenhague"), ("la Finlande", "de la Finlande", "Helsinki"),
    ("l'Autriche", "de l'Autriche", "Vienne"), ("la Hongrie", "de la Hongrie", "Budapest"),
    ("la Roumanie", "de la Roumanie", "Bucarest"), ("l'Ukraine", "de l'Ukraine", "Kiev"),
    ("l'Arabie saoudite", "de l'Arabie saoudite", "Riyad"), ("l'Iran", "de l'Iran", "Téhéran"),
    ("l'Irak", "de l'Irak", "Bagdad"), ("le Vietnam", "du Vietnam", "Hanoï"),
    ("la Thaïlande", "de la Thaïlande", "Bangkok"), ("l'Indonésie", "de l'Indonésie", "Jakarta"),
    ("le Bénin", "du Bénin", "Porto-Novo"), ("le Togo", "du Togo", "Lomé"),
    ("le Burkina Faso", "du Burkina Faso", "Ouagadougou"), ("la Guinée", "de la Guinée", "Conakry"),
    ("Madagascar", "de Madagascar", "Antananarivo"), ("le Rwanda", "du Rwanda", "Kigali"),
    ("l'Angola", "de l'Angola", "Luanda"), ("la Zambie", "de la Zambie", "Lusaka"),
    ("la Libye", "de la Libye", "Tripoli"), ("la Mauritanie", "de la Mauritanie", "Nouakchott"),
    ("la République centrafricaine", "de la République centrafricaine", "Bangui"),
]

DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
          "septembre", "octobre", "novembre", "décembre"]
_VOWELS = "aeiouhéèêàâîïôûAEIOUH"

# (question, réponse) — faits simples et sûrs
FACTS = [
    ("Combien de jours y a-t-il dans une semaine ?", "Il y a sept jours dans une semaine."),
    ("Combien de mois y a-t-il dans une année ?", "Il y a douze mois dans une année."),
    ("Combien de minutes y a-t-il dans une heure ?", "Il y a soixante minutes dans une heure."),
    ("Combien de secondes y a-t-il dans une minute ?", "Il y a soixante secondes dans une minute."),
    ("Combien d'heures y a-t-il dans une journée ?", "Il y a vingt-quatre heures dans une journée."),
    ("Combien de jours compte une année normale ?", "Une année normale compte 365 jours."),
    ("Combien de lettres y a-t-il dans l'alphabet français ?", "L'alphabet français compte 26 lettres."),
    ("Combien de côtés a un triangle ?", "Un triangle a trois côtés."),
    ("Combien de côtés a un carré ?", "Un carré a quatre côtés."),
    ("Combien de pattes a une araignée ?", "Une araignée a huit pattes."),
    ("Combien de planètes compte le système solaire ?", "Le système solaire compte huit planètes."),
    ("Quelle est la plus grande planète du système solaire ?", "Jupiter est la plus grande planète du système solaire."),
    ("Quelle est la planète la plus proche du Soleil ?", "Mercure est la planète la plus proche du Soleil."),
    ("Autour de quoi la Terre tourne-t-elle ?", "La Terre tourne autour du Soleil."),
    ("De quelle couleur est le ciel par temps clair ?", "Par temps clair, le ciel est bleu."),
    ("Quelle est la formule chimique de l'eau ?", "La formule chimique de l'eau est H2O."),
    ("À quelle température l'eau bout-elle ?", "L'eau bout à 100 degrés Celsius au niveau de la mer."),
    ("À quelle température l'eau gèle-t-elle ?", "L'eau gèle à 0 degré Celsius."),
    ("Quel est le plus grand océan du monde ?", "L'océan Pacifique est le plus grand océan du monde."),
    ("Quel est le plus haut sommet du monde ?", "Le mont Everest est le plus haut sommet du monde."),
    ("Quel est le plus long fleuve d'Afrique ?", "Le Nil est le plus long fleuve d'Afrique."),
    ("Quel est le plus grand désert chaud du monde ?", "Le Sahara est le plus grand désert chaud du monde."),
    ("Quelle est la langue officielle de la France ?", "La langue officielle de la France est le français."),
    ("Quelles sont les langues officielles du Cameroun ?", "Le Cameroun a deux langues officielles : le français et l'anglais."),
    ("Quelle est la plus grande ville du Cameroun ?", "Douala est la plus grande ville du Cameroun."),
    ("Quelle est la monnaie du Cameroun ?", "La monnaie du Cameroun est le franc CFA."),
    ("Qui a écrit Les Misérables ?", "Victor Hugo a écrit Les Misérables."),
    ("Qui a peint la Joconde ?", "Léonard de Vinci a peint la Joconde."),
    ("Quel animal miaule ?", "Le chat miaule."),
    ("Quel animal aboie ?", "Le chien aboie."),
    ("Quel animal donne du lait et fait « meuh » ?", "La vache donne du lait et fait « meuh »."),
    ("Que boivent les vaches ?", "Les vaches boivent de l'eau."),
    ("Que produisent les abeilles ?", "Les abeilles produisent du miel."),
    ("De quoi les plantes ont-elles besoin pour grandir ?", "Les plantes ont besoin d'eau, de lumière et d'air pour grandir."),
    ("Quelle est la saison la plus froide ?", "L'hiver est la saison la plus froide."),
    ("Combien de saisons y a-t-il dans une année ?", "Il y a quatre saisons dans une année : le printemps, l'été, l'automne et l'hiver."),
    ("Comment appelle-t-on le petit d'un chien ?", "Le petit d'un chien s'appelle un chiot."),
    ("Comment appelle-t-on le petit d'un chat ?", "Le petit d'un chat s'appelle un chaton."),
    ("Comment appelle-t-on le petit de la vache ?", "Le petit de la vache s'appelle un veau."),
    ("Avec quoi voit-on ?", "On voit avec les yeux."),
    ("Avec quoi entend-on ?", "On entend avec les oreilles."),
    ("Combien de doigts a une main ?", "Une main a cinq doigts."),
]

OPPOSITES = [("chaud", "froid"), ("grand", "petit"), ("jour", "nuit"), ("haut", "bas"), ("lent", "rapide"),
             ("ouvert", "fermé"), ("jeune", "vieux"), ("plein", "vide"), ("léger", "lourd"), ("fort", "faible"),
             ("blanc", "noir"), ("heureux", "triste"), ("facile", "difficile"), ("propre", "sale"), ("près", "loin")]

# Questions hors de portée : réponse honnête, courte, en phrase
CANNOT = [
    ("Quel temps fera-t-il demain ?", "Je ne peux pas connaître la météo, je suis un tout petit modèle de langage."),
    ("Quelle heure est-il ?", "Je n'ai pas accès à l'heure, désolé."),
    ("Quel jour sommes-nous aujourd'hui ?", "Je n'ai pas accès à la date du jour, désolé."),
    ("Quel est mon nom ?", "Je ne connais pas ton nom. Tu peux me le dire !"),
    ("Où est-ce que j'habite ?", "Je ne sais pas où tu habites."),
    ("Peux-tu naviguer sur Internet ?", "Non, je ne peux pas naviguer sur Internet."),
    ("Peux-tu m'appeler au téléphone ?", "Non, je ne peux pas passer d'appels. Je peux seulement répondre par écrit."),
    ("Quel sera le résultat du prochain match ?", "Je ne peux pas prédire l'avenir, désolé."),
    ("Combien j'ai d'argent sur mon compte ?", "Je n'ai aucun accès à tes comptes."),
    ("Peux-tu voir ma photo ?", "Non, je ne peux lire que du texte."),
]

IDENTITY = [
    (["Qui es-tu ?", "Tu es qui ?", "Présente-toi.", "Peux-tu te présenter ?", "Tu es quoi ?"],
     [f"Je suis {BOT_NAME}, un tout petit modèle de langage créé par {BOT_MAKER}.",
      f"Je m'appelle {BOT_NAME}. Je suis un petit modèle de langage créé par {BOT_MAKER}."]),
    (["Comment tu t'appelles ?", "Quel est ton nom ?", "Comment t'appelles-tu ?", "Tu t'appelles comment ?"],
     [f"Je m'appelle {BOT_NAME}.", f"Mon nom est {BOT_NAME}."]),
    (["Qui t'a créé ?", "Qui t'a fabriqué ?", "Qui est ton créateur ?", "Qui t'a programmé ?"],
     [f"J'ai été créé par {BOT_MAKER}.", f"C'est {BOT_MAKER} qui m'a créé."]),
    (["Es-tu un humain ?", "Tu es une personne ?", "Es-tu une vraie personne ?"],
     ["Non, je ne suis pas un humain. Je suis un petit modèle de langage."]),
    (["Que sais-tu faire ?", "À quoi sers-tu ?", "Que peux-tu faire ?", "Tu peux m'aider ?"],
     ["Je peux répondre à des questions simples en français, mais je reste un tout petit modèle : je peux me tromper.",
      "J'essaie de répondre à des questions simples en français. Je suis très petit, alors je fais parfois des erreurs."]),
    (["Quelle langue parles-tu ?", "Tu parles quelle langue ?", "Parles-tu français ?"],
     ["Je parle français."]),
    (["Es-tu intelligent ?", "Tu es intelligent ?"],
     ["Je suis un tout petit modèle : je fais de mon mieux, mais je ne sais pas tout."]),
]

GREETINGS = [
    (["Bonjour", "Bonjour !", "Salut", "Salut !", "Coucou", "Hello", "Bonjour à toi"],
     ["Bonjour ! Comment puis-je t'aider ?", "Salut ! Que puis-je faire pour toi ?", "Bonjour ! Je t'écoute."]),
    (["Bonsoir", "Bonsoir !"], ["Bonsoir ! Comment puis-je t'aider ?"]),
    (["Comment ça va ?", "Ça va ?", "Comment vas-tu ?", "Tu vas bien ?"],
     ["Je vais bien, merci ! Et toi ?", "Ça va bien, merci de demander. Et toi ?"]),
    (["Merci", "Merci !", "Merci beaucoup", "Merci bien"], ["Avec plaisir !", "De rien !", "Je t'en prie !"]),
    (["Au revoir", "À bientôt", "Bonne journée", "À demain", "Bonne nuit"],
     ["Au revoir ! À bientôt.", "À bientôt !", "Bonne journée à toi aussi !"]),
    (["Ça va bien", "Je vais bien", "Très bien, merci"], ["Tant mieux ! Que puis-je faire pour toi ?"]),
]


def _capital_examples(rng: random.Random) -> List[Dict]:
    out = []
    q_tpl = ["Quelle est la capitale {de} ?", "C'est quoi la capitale {de} ?", "Peux-tu me dire la capitale {de} ?",
             "Dis-moi la capitale {de}.", "La capitale {de}, c'est quoi ?", "Quelle est la capitale {de} ?"]
    a_tpl = ["La capitale {de} est {cap}.", "{cap} est la capitale {de}.", "La capitale {de} est {cap}."]
    for name, de, cap in CAPITALS:
        for q in rng.sample(q_tpl, 3):
            out.append(_conv(q.format(de=de), rng.choice(a_tpl).format(de=de, cap=cap)))
        # question inverse
        out.append(_conv(f"{cap} est la capitale de quel pays ?", f"{cap} est la capitale {de}."))
    return out


def _conv(q: str, a: str) -> Dict:
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def _arithmetic_examples(rng: random.Random, n: int) -> List[Dict]:
    out = []
    for _ in range(n):
        kind = rng.choice(["add", "add", "sub", "mul"])
        if kind == "add":
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            r, word, sym = a + b, "plus", "+"
        elif kind == "sub":
            a, b = rng.randint(0, 60), rng.randint(0, 60)
            a, b = max(a, b), min(a, b)
            r, word, sym = a - b, "moins", "-"
        else:
            a, b = rng.randint(0, 10), rng.randint(0, 10)
            r, word, sym = a * b, "fois", "×"
        if rng.random() < 0.6:
            q = rng.choice([f"Combien font {a} {word} {b} ?", f"Combien fait {a} {word} {b} ?", f"{a} {word} {b} ?"])
            ans = f"{a} {word} {b} font {r}." if r != 1 else f"{a} {word} {b} fait 1."
        else:
            q = f"Combien font {a} {sym} {b} ?"
            ans = f"{a} {sym} {b} = {r}."
        out.append(_conv(q, ans))
    return out


def _calendar_examples(rng: random.Random) -> List[Dict]:
    out = []
    for i, d in enumerate(DAYS):
        nxt, prev = DAYS[(i + 1) % 7], DAYS[(i - 1) % 7]
        out.append(_conv(f"Quel jour vient après le {d} ?", f"Après le {d}, c'est le {nxt}."))
        out.append(_conv(f"Quel jour vient avant le {d} ?", f"Avant le {d}, c'est le {prev}."))
    for i, m in enumerate(MONTHS):
        nxt, prev = MONTHS[(i + 1) % 12], MONTHS[(i - 1) % 12]
        art = "d'" if m[0] in _VOWELS else "de "
        out.append(_conv(f"Quel mois vient après {m} ?", f"Après {m}, c'est {nxt}."))
        out.append(_conv(f"Quel mois vient avant {m} ?", f"Avant {m}, c'est {prev}."))
        out.append(_conv(f"Quel est le numéro du mois {art}{m} ?", f"{m.capitalize()} est le mois numéro {i + 1}."))
    out.append(_conv("Cite les jours de la semaine.", "Les jours de la semaine sont : " + ", ".join(DAYS[:-1]) + " et " + DAYS[-1] + "."))
    out.append(_conv("Cite les mois de l'année.", "Les mois de l'année sont : " + ", ".join(MONTHS[:-1]) + " et " + MONTHS[-1] + "."))
    return out


def _opposite_examples(rng: random.Random) -> List[Dict]:
    out = []
    for a, b in OPPOSITES:
        out.append(_conv(f"Quel est le contraire de {a} ?", f"Le contraire de {a} est {b}."))
        out.append(_conv(f"Quel est le contraire de {b} ?", f"Le contraire de {b} est {a}."))
    return out


def _pairs(table, rng: random.Random, reps: int) -> List[Dict]:
    out = []
    for questions, answers in table:
        for _ in range(reps):
            out.append(_conv(rng.choice(questions), rng.choice(answers)))
    return out


def build_synthetic_qa(seed: int = 0, n_arithmetic: int = 1200, id_reps: int = 8, two_turn_frac: float = 0.15) -> List[Dict]:
    """Renvoie une liste de conversations {"messages": [...]} mélangées (déterministe pour un seed donné)."""
    rng = random.Random(seed)
    single: List[Dict] = []
    single += _capital_examples(rng)
    single += _arithmetic_examples(rng, n_arithmetic)
    single += _calendar_examples(rng)
    single += _opposite_examples(rng)
    single += [_conv(q, a) for q, a in FACTS for _ in range(3)]
    single += [_conv(q, a) for q, a in CANNOT for _ in range(3)]
    single += _pairs(IDENTITY, rng, id_reps)
    single += _pairs(GREETINGS, rng, id_reps)
    rng.shuffle(single)

    # conversations à 2 tours : on enchaîne deux échanges indépendants (apprend l'alternance des tours)
    n_two = int(len(single) * two_turn_frac)
    two_turn = []
    for _ in range(n_two):
        a, b = rng.sample(single, 2)
        two_turn.append({"messages": a["messages"] + b["messages"]})
    data = single + two_turn
    rng.shuffle(data)
    return data


if __name__ == "__main__":
    d = build_synthetic_qa()
    print(len(d), "conversations")
    for ex in d[:8]:
        print(ex["messages"])
