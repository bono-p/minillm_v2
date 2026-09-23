"""
evaluate.py — mesures de qualité + batterie QA ouverte.

Modes :
  perplexity : perplexité sur le validation set du pré-entraînement.
  qa         : évaluation quantitative sur le val SFT avec attendu.
  demo       : batterie de questions ouvertes sans attendu.
  persona    : vérification de la personnalité.

La batterie UNSEEN_QA_PROMPTS est volontairement indépendante
du dataset SFT : elle sert à tester le comportement général du modèle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from typing import Dict, List

import numpy as np
import torch

from data import PretrainData
from generate import chat_reply, load_model, stream_tokens


# ============================================================
# QUESTIONS OUVERTES — aucun attendu
# ============================================================
#
# Ces questions sont destinées à tester :
# - connaissances générales
# - français
# - logique
# - calcul
# - sciences
# - géographie
# - histoire
# - informatique
# - programmation
# - raisonnement
# - compréhension
# - génération
#
# Elles ne sont PAS utilisées pour calculer EM/F1.
#

UNSEEN_QA_PROMPTS = [

    # --------------------------------------------------------
    # CONVERSATION / BASE
    # --------------------------------------------------------
    "Bonjour.",
    "Salut, comment vas-tu ?",
    "Qui es-tu ?",
    "Comment t'appelles-tu ?",
    "Que peux-tu faire ?",
    "Que ne peux-tu pas faire ?",
    "Pourquoi les gens utilisent-ils des assistants IA ?",
    "Explique-moi simplement ce qu'est une intelligence artificielle.",
    "Quelle est la différence entre une IA et un programme classique ?",
    "Qu'est-ce qu'un modèle de langage ?",
    "Qu'est-ce qu'un chatbot ?",
    "Pourquoi un modèle de langage peut-il se tromper ?",
    "Pourquoi une IA peut-elle inventer une information ?",
    "Qu'est-ce qu'une hallucination dans une IA ?",

    # --------------------------------------------------------
    # FRANÇAIS
    # --------------------------------------------------------
    "Quel est le contraire de grand ?",
    "Quel est le contraire de rapide ?",
    "Quel est le contraire de difficile ?",
    "Quel est le synonyme de heureux ?",
    "Quel est le synonyme de commencer ?",
    "Quel est le pluriel de cheval ?",
    "Quel est le pluriel de journal ?",
    "Quel est le féminin de acteur ?",
    "Quel est le féminin de directeur ?",
    "Quelle est la différence entre a et à ?",
    "Quelle est la différence entre et et est ?",
    "Quelle est la différence entre son et sont ?",
    "Quelle est la différence entre ces et ses ?",
    "Corrige cette phrase : Le garçon mange une pomme.",
    "Corrige cette phrase : Les enfant joue dehors.",
    "Corrige cette phrase : Elle a été très fatigué hier.",
    "Corrige cette phrase : Nous somme allé au marché.",
    "Explique ce qu'est un verbe.",
    "Explique ce qu'est un nom commun.",
    "Explique ce qu'est un adjectif.",
    "Explique ce qu'est une phrase.",
    "Qu'est-ce qu'un synonyme ?",
    "Qu'est-ce qu'un antonyme ?",
    "Qu'est-ce qu'une métaphore ?",
    "Qu'est-ce qu'une comparaison ?",

    # --------------------------------------------------------
    # MATHÉMATIQUES — CALCUL SIMPLE
    # --------------------------------------------------------
    "Combien font 2 plus 3 ?",
    "Combien font 7 plus 8 ?",
    "Combien font 12 plus 19 ?",
    "Combien font 25 plus 37 ?",
    "Combien font 100 moins 37 ?",
    "Combien font 81 moins 29 ?",
    "Combien font 7 fois 8 ?",
    "Combien font 12 fois 12 ?",
    "Combien font 144 divisé par 12 ?",
    "Combien font 100 divisé par 4 ?",
    "Combien font 15 fois 6 ?",
    "Combien font 125 plus 375 ?",
    "Combien font 1000 moins 275 ?",
    "Combien font 23 plus 48 moins 11 ?",
    "Combien font 7 fois 9 plus 3 ?",
    "Combien font 50 pour cent de 200 ?",
    "Combien font 10 pour cent de 500 ?",
    "Combien font 25 pour cent de 80 ?",
    "Quel est le carré de 12 ?",
    "Quel est le carré de 15 ?",
    "Quelle est la racine carrée de 81 ?",
    "Quelle est la racine carrée de 144 ?",
    "Combien font 2 puissance 5 ?",
    "Combien font 2 puissance 10 ?",

    # --------------------------------------------------------
    # MATHS — RAISONNEMENT
    # --------------------------------------------------------
    "Si j'ai 10 pommes et que j'en donne 3, combien m'en reste-t-il ?",
    "Si un livre coûte 2000 francs et que j'en achète 3, combien dois-je payer ?",
    "Si une voiture parcourt 60 km en une heure, quelle distance parcourt-elle en 3 heures ?",
    "Si 5 personnes se partagent équitablement 20 bonbons, combien chaque personne reçoit-elle ?",
    "Un train part à 8 heures et arrive à 11 heures. Combien de temps dure le trajet ?",
    "Si 4 stylos coûtent 1000 francs, combien coûtent 8 stylos ?",
    "Une classe contient 30 élèves. 12 sont des filles. Combien y a-t-il de garçons ?",
    "J'ai 5000 francs et je dépense 1750 francs. Combien me reste-t-il ?",
    "Une bouteille contient 1,5 litre. Combien contiennent 4 bouteilles ?",
    "Si un nombre est multiplié par 5 et donne 40, quel est ce nombre ?",

    # --------------------------------------------------------
    # LOGIQUE
    # --------------------------------------------------------
    "Quel nombre vient après 1, 2, 3, 4 ?",
    "Quel nombre vient après 2, 4, 6, 8 ?",
    "Quel nombre vient après 5, 10, 15, 20 ?",
    "Quel nombre manque : 2, 4, 6, ?, 10 ?",
    "Quel nombre manque : 5, 10, ?, 20, 25 ?",
    "Quel nombre manque : 1, 3, 5, ?, 9 ?",
    "Quel est l'intrus entre pomme, banane, carotte et orange ?",
    "Quel est l'intrus entre chien, chat, cheval et voiture ?",
    "Si tous les chats sont des animaux et que Mimi est un chat, que peut-on dire de Mimi ?",
    "Si A est plus grand que B et B est plus grand que C, lequel est le plus petit ?",
    "Paul est plus âgé que Jean. Jean est plus âgé que Marc. Qui est le plus jeune ?",
    "Si aujourd'hui est lundi, quel jour sera demain ?",
    "Si aujourd'hui est mercredi, quel jour sera après-demain ?",
    "Quel jour vient après vendredi ?",
    "Combien de jours y a-t-il dans une semaine ?",
    "Combien de mois y a-t-il dans une année ?",
    "Combien d'heures y a-t-il dans une journée ?",
    "Combien de minutes y a-t-il dans une heure ?",
    "Combien de secondes y a-t-il dans une minute ?",

    # --------------------------------------------------------
    # SCIENCES
    # --------------------------------------------------------
    "Qu'est-ce que la photosynthèse ?",
    "Pourquoi les plantes ont-elles besoin de lumière ?",
    "Pourquoi le ciel est-il bleu ?",
    "Pourquoi la pluie tombe-t-elle ?",
    "Qu'est-ce que l'eau ?",
    "Quels sont les trois états principaux de la matière ?",
    "Quelle est la différence entre solide, liquide et gaz ?",
    "Qu'est-ce que l'évaporation ?",
    "Qu'est-ce que la condensation ?",
    "Qu'est-ce que la gravité ?",
    "Pourquoi les objets tombent-ils vers le sol ?",
    "Qu'est-ce qu'un atome ?",
    "Qu'est-ce qu'une molécule ?",
    "Qu'est-ce qu'une cellule ?",
    "Quelle est la fonction principale du cœur ?",
    "Quel est le rôle des poumons ?",
    "Pourquoi avons-nous besoin d'oxygène ?",
    "Qu'est-ce que l'ADN ?",
    "Qu'est-ce qu'une bactérie ?",
    "Qu'est-ce qu'un virus ?",
    "Quelle est la différence entre une plante et un animal ?",
    "Pourquoi les humains ont-ils besoin de dormir ?",
    "Pourquoi avons-nous besoin de boire de l'eau ?",
    "Qu'est-ce que l'énergie ?",
    "Qu'est-ce que l'électricité ?",
    "Qu'est-ce qu'un aimant ?",
    "Qu'est-ce que la lumière ?",
    "Quelle est la vitesse approximative de la lumière ?",

    # --------------------------------------------------------
    # TERRE / ESPACE
    # --------------------------------------------------------
    "Quelle est la planète sur laquelle nous vivons ?",
    "Combien y a-t-il de planètes dans le système solaire ?",
    "Quelle est la planète la plus proche du Soleil ?",
    "Quelle est la plus grande planète du système solaire ?",
    "Quelle est la planète surnommée la planète rouge ?",
    "Qu'est-ce que le Soleil ?",
    "Qu'est-ce que la Lune ?",
    "Pourquoi y a-t-il des phases de la Lune ?",
    "Pourquoi y a-t-il le jour et la nuit ?",
    "Pourquoi y a-t-il des saisons ?",
    "Qu'est-ce qu'une étoile ?",
    "Qu'est-ce qu'une galaxie ?",
    "Dans quelle galaxie se trouve le système solaire ?",
    "Qu'est-ce qu'une éclipse solaire ?",
    "Qu'est-ce qu'une éclipse lunaire ?",
    "Qu'est-ce qu'un trou noir ?",
    "Qu'est-ce qu'une planète ?",
    "Quelle est la différence entre une étoile et une planète ?",

    # --------------------------------------------------------
    # GÉOGRAPHIE
    # --------------------------------------------------------
    "Quelle est la capitale du Cameroun ?",
    "Quelle est la plus grande ville du Cameroun ?",
    "Dans quelle région se trouve Garoua ?",
    "Quel est le plus grand pays d'Afrique par superficie ?",
    "Quel est le plus petit pays d'Afrique par superficie ?",
    "Combien de pays compte approximativement l'Afrique ?",
    "Quel est le plus grand océan du monde ?",
    "Quel est le plus grand continent du monde ?",
    "Quelle est la capitale de la France ?",
    "Quelle est la capitale du Nigeria ?",
    "Quelle est la capitale du Tchad ?",
    "Quelle est la capitale de la République centrafricaine ?",
    "Quelle est la capitale du Gabon ?",
    "Quelle est la capitale de la Côte d'Ivoire ?",
    "Quelle est la capitale du Sénégal ?",
    "Quelle est la capitale du Ghana ?",
    "Quelle est la capitale du Kenya ?",
    "Quelle est la capitale de l'Égypte ?",
    "Quel fleuve traverse l'Égypte ?",
    "Quel est le plus long fleuve d'Afrique ?",
    "Qu'est-ce qu'un désert ?",
    "Qu'est-ce qu'une montagne ?",
    "Qu'est-ce qu'un océan ?",

    # --------------------------------------------------------
    # HISTOIRE / CULTURE GÉNÉRALE
    # --------------------------------------------------------
    "Qu'est-ce que l'histoire ?",
    "Qu'est-ce qu'une civilisation ?",
    "Qu'est-ce qu'un empire ?",
    "Qu'est-ce qu'une révolution ?",
    "Qu'est-ce qu'une démocratie ?",
    "Qu'est-ce qu'une constitution ?",
    "Qu'est-ce que l'indépendance d'un pays ?",
    "Quand le Cameroun est-il devenu indépendant ?",
    "Qui était Nelson Mandela ?",
    "Qui était Albert Einstein ?",
    "Qui était Marie Curie ?",
    "Qui était Isaac Newton ?",
    "Qui était Léonard de Vinci ?",
    "Qu'est-ce que la Renaissance ?",
    "Qu'est-ce que la révolution industrielle ?",
    "Qu'est-ce que l'Antiquité ?",
    "Qu'est-ce que le Moyen Âge ?",

    # --------------------------------------------------------
    # INFORMATIQUE
    # --------------------------------------------------------
    "Qu'est-ce qu'un ordinateur ?",
    "Qu'est-ce qu'un processeur ?",
    "Qu'est-ce que la mémoire RAM ?",
    "À quoi sert un disque SSD ?",
    "Quelle est la différence entre RAM et stockage ?",
    "Qu'est-ce qu'un système d'exploitation ?",
    "Qu'est-ce que Linux ?",
    "Qu'est-ce que Windows ?",
    "Qu'est-ce qu'Android ?",
    "Qu'est-ce qu'un fichier ?",
    "Qu'est-ce qu'un dossier informatique ?",
    "Qu'est-ce qu'un serveur ?",
    "Qu'est-ce qu'un client en informatique ?",
    "Qu'est-ce qu'une adresse IP ?",
    "Qu'est-ce qu'Internet ?",
    "Qu'est-ce qu'un navigateur web ?",
    "Qu'est-ce que HTTP ?",
    "Qu'est-ce que HTTPS ?",
    "Qu'est-ce qu'une API ?",
    "Qu'est-ce qu'une base de données ?",
    "Qu'est-ce que SQL ?",
    "Qu'est-ce que JSON ?",
    "Qu'est-ce que Git ?",
    "Qu'est-ce que GitHub ?",
    "Qu'est-ce que le cloud computing ?",
    "Qu'est-ce qu'un conteneur Docker ?",
    "Qu'est-ce qu'une machine virtuelle ?",
    "Qu'est-ce qu'un compilateur ?",
    "Qu'est-ce qu'un interpréteur ?",
    "Quelle est la différence entre Python et JavaScript ?",

    # --------------------------------------------------------
    # PROGRAMMATION
    # --------------------------------------------------------
    "Qu'est-ce qu'une variable en programmation ?",
    "Qu'est-ce qu'une fonction ?",
    "Qu'est-ce qu'une boucle ?",
    "Qu'est-ce qu'une condition ?",
    "Qu'est-ce qu'une liste en Python ?",
    "Qu'est-ce qu'un dictionnaire en Python ?",
    "Qu'est-ce qu'une classe en programmation ?",
    "Qu'est-ce qu'un objet en programmation ?",
    "Qu'est-ce que la programmation orientée objet ?",
    "Qu'est-ce qu'une exception en Python ?",
    "À quoi sert try except en Python ?",
    "Qu'est-ce qu'un module Python ?",
    "Qu'est-ce qu'un package Python ?",
    "Qu'est-ce que pip ?",
    "Qu'est-ce qu'un environnement virtuel Python ?",
    "Qu'est-ce qu'un algorithme ?",
    "Qu'est-ce que la complexité algorithmique ?",
    "Qu'est-ce que O(n) ?",
    "Qu'est-ce qu'une API REST ?",
    "Quelle est la différence entre GET et POST ?",

    # --------------------------------------------------------
    # IA / MACHINE LEARNING
    # --------------------------------------------------------
    "Qu'est-ce que le machine learning ?",
    "Qu'est-ce que le deep learning ?",
    "Quelle est la différence entre IA et machine learning ?",
    "Qu'est-ce qu'un réseau de neurones ?",
    "Qu'est-ce qu'un neurone artificiel ?",
    "Qu'est-ce qu'une fonction d'activation ?",
    "Qu'est-ce qu'une fonction de perte ?",
    "Qu'est-ce que l'entraînement d'un modèle ?",
    "Qu'est-ce qu'un dataset ?",
    "Qu'est-ce qu'un batch ?",
    "Qu'est-ce qu'une époque d'entraînement ?",
    "Qu'est-ce que le surapprentissage ?",
    "Qu'est-ce que le sous-apprentissage ?",
    "Qu'est-ce que la validation d'un modèle ?",
    "Qu'est-ce qu'un jeu de test ?",
    "Qu'est-ce qu'un hyperparamètre ?",
    "Qu'est-ce que le learning rate ?",
    "Qu'est-ce qu'un checkpoint ?",
    "Qu'est-ce qu'un tokenizer ?",
    "Pourquoi un LLM utilise-t-il des tokens ?",
    "Qu'est-ce qu'un modèle autoregressif ?",
    "Comment un modèle de langage prédit-il le prochain token ?",
    "Qu'est-ce qu'un Transformer ?",
    "Qu'est-ce que l'attention dans un Transformer ?",
    "Qu'est-ce que l'attention causale ?",
    "Qu'est-ce que le fine-tuning ?",
    "Qu'est-ce que le SFT ?",
    "Qu'est-ce que le RAG ?",
    "Qu'est-ce qu'un Mixture of Experts ?",
    "Pourquoi augmenter le nombre de paramètres d'un modèle ?",

    # --------------------------------------------------------
    # RAISONNEMENT / EXPLICATION
    # --------------------------------------------------------
    "Pourquoi le feu produit-il de la chaleur ?",
    "Pourquoi la glace flotte-t-elle sur l'eau ?",
    "Pourquoi le métal paraît-il froid ?",
    "Pourquoi une ombre apparaît-elle derrière un objet ?",
    "Pourquoi une balle finit-elle par s'arrêter lorsqu'on la fait rouler ?",
    "Pourquoi les pneus d'une voiture sont-ils en caoutchouc ?",
    "Pourquoi les avions peuvent-ils voler ?",
    "Pourquoi les bateaux flottent-ils ?",
    "Pourquoi une ampoule produit-elle de la lumière ?",
    "Pourquoi une plante a-t-elle besoin de racines ?",
    "Pourquoi les feuilles sont-elles généralement vertes ?",
    "Pourquoi les humains ont-ils deux yeux ?",
    "Pourquoi avons-nous deux oreilles ?",
    "Pourquoi faut-il se laver les mains ?",
    "Pourquoi les aliments doivent-ils parfois être conservés au réfrigérateur ?",

    # --------------------------------------------------------
    # VIE QUOTIDIENNE
    # --------------------------------------------------------
    "Comment faire bouillir de l'eau ?",
    "Comment préparer du riz simplement ?",
    "Comment fonctionne un réfrigérateur ?",
    "Comment fonctionne une batterie ?",
    "Comment fonctionne un téléphone portable ?",
    "Comment fonctionne une lampe de poche ?",
    "Comment fonctionne une connexion Wi-Fi ?",
    "Comment fonctionne le GPS ?",
    "Comment fonctionne un moteur de voiture ?",
    "Comment fonctionne une serrure ?",

    # --------------------------------------------------------
    # QUESTIONS PIÈGES / ROBUSTESSE
    # --------------------------------------------------------
    "Peux-tu répondre à une question dont tu ne connais pas la réponse ?",
    "Que dois-tu faire si tu n'es pas certain d'une information ?",
    "Est-il préférable d'inventer une réponse ou de dire qu'on ne sait pas ?",
    "Peux-tu distinguer un fait d'une opinion ?",
    "Quelle est la différence entre savoir quelque chose et faire une supposition ?",
    "Si une question contient une information fausse, dois-tu forcément l'accepter ?",
    "Que signifie vérifier une information ?",
    "Pourquoi deux personnes peuvent-elles avoir des opinions différentes ?",
    "Une réponse très longue est-elle forcément meilleure qu'une réponse courte ?",
    "Une réponse confiante est-elle forcément vraie ?",

    # --------------------------------------------------------
    # CRÉATIVITÉ / GÉNÉRATION
    # --------------------------------------------------------
    "Raconte-moi une courte histoire de cinq phrases.",
    "Écris un petit poème sur la pluie.",
    "Écris une histoire avec un robot et un étudiant.",
    "Invente une devinette simple.",
    "Invente une devinette difficile.",
    "Raconte une blague courte.",
    "Explique les mathématiques comme si j'avais cinq ans.",
    "Explique Internet comme si j'avais cinq ans.",
    "Explique l'intelligence artificielle à un enfant.",
    "Écris une courte description d'une ville africaine imaginaire.",

    # --------------------------------------------------------
    # COMPARAISONS
    # --------------------------------------------------------
    "Quelle est la différence entre une voiture et une moto ?",
    "Quelle est la différence entre un téléphone et un ordinateur ?",
    "Quelle est la différence entre Internet et le Web ?",
    "Quelle est la différence entre une base de données et un fichier JSON ?",
    "Quelle est la différence entre entraînement et inférence ?",
    "Quelle est la différence entre pré-entraînement et fine-tuning ?",
    "Quelle est la différence entre CPU et GPU ?",
    "Quelle est la différence entre RAM et VRAM ?",
    "Quelle est la différence entre logiciel et matériel ?",
    "Quelle est la différence entre un serveur et un ordinateur personnel ?",

    # --------------------------------------------------------
    # QUESTIONS MULTI-ÉTAPES
    # --------------------------------------------------------
    "Si j'ai 10000 francs, que je dépense 2500 francs puis 1750 francs, combien me reste-t-il ?",
    "Un magasin vend 3 cahiers à 500 francs chacun. Combien coûtent-ils au total ?",
    "Un étudiant étudie 2 heures le matin et 3 heures le soir pendant 5 jours. Combien d'heures étudie-t-il ?",
    "Une voiture consomme 8 litres pour 100 km. Combien consomme-t-elle approximativement pour 300 km ?",
    "Une classe de 40 élèves est divisée en 5 groupes égaux. Combien d'élèves y a-t-il dans chaque groupe ?",
    "Un serveur reçoit 100 requêtes par seconde. Combien de requêtes reçoit-il en 10 secondes ?",
    "Un modèle traite 1000 tokens par seconde. Combien de tokens peut-il traiter en une minute ?",
    "Un dataset contient 1 million de tokens et un entraînement utilise 3 epochs. Combien de tokens sont vus au total ?",
    "Un modèle traite 65536 tokens par itération. Combien de tokens traite-t-il en 1000 itérations ?",
    "Si un modèle a 10 millions de paramètres et qu'un autre en a 50 millions, combien de fois le second possède-t-il plus de paramètres ?",

    # --------------------------------------------------------
    # QUESTIONS OUVERTES
    # --------------------------------------------------------
    "Pourquoi apprend-on les mathématiques ?",
    "Pourquoi apprend-on à programmer ?",
    "Pourquoi les langues existent-elles ?",
    "Pourquoi les humains construisent-ils des villes ?",
    "Pourquoi les sociétés ont-elles besoin de règles ?",
    "Pourquoi les ordinateurs utilisent-ils des nombres binaires ?",
    "Pourquoi les modèles de langage ont-ils besoin de beaucoup de données ?",
    "Pourquoi un petit modèle peut-il parfois répondre correctement à une question ?",
    "Pourquoi un modèle plus grand n'est-il pas nécessairement parfait ?",
    "Quels sont les avantages d'un petit modèle de langage ?",
    "Quels sont les inconvénients d'un petit modèle de langage ?",
    "Pourquoi quantifier un modèle d'intelligence artificielle ?",
    "Pourquoi entraîner un modèle sur plusieurs époques ?",
    "Pourquoi séparer les données d'entraînement et de validation ?",
]


# Garder les anciens prompts simples si tu veux les utiliser séparément.
DEMO_PROMPTS = [
    "Bonjour",
    "Qui es-tu ?",
    "Comment tu t'appelles ?",
    "Qui t'a créé ?",
    "Quelle est la capitale du Cameroun ?",
    "Combien font 12 plus 7 ?",
    "Quel jour vient après le mardi ?",
    "Quel est le contraire de grand ?",
    "Combien de jours y a-t-il dans une semaine ?",
    "Explique ce qu'est la photosynthèse.",
    "Raconte-moi une blague.",
]


_ARTICLES = {
    "le", "la", "les", "l", "un", "une", "des",
    "du", "de", "d", "au", "aux"
}


def normalize(s: str) -> List[str]:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    return [t for t in s.split() if t not in _ARTICLES]


def f1_score(pred: str, ref: str) -> float:
    p, r = normalize(pred), normalize(ref)

    if not p or not r:
        return float(p == r)

    common = {}

    for t in p:
        if t in r:
            common[t] = min(p.count(t), r.count(t))

    n = sum(common.values())

    if n == 0:
        return 0.0

    prec = n / len(p)
    rec = n / len(r)

    return 2 * prec * rec / (prec + rec)


def perplexity(
    ckpt: str,
    data_dir: str,
    seq_len: int = 512,
    n_batches: int = 200,
    batch_size: int = 16,
    device: str = "auto"
) -> Dict:

    from train import evaluate
    import contextlib

    model, tok, meta = load_model(ckpt, device=device)

    data = PretrainData(
        data_dir,
        seq_len,
        batch_size
    )

    dev = next(model.parameters()).device

    loss, n = evaluate(
        model,
        data.val_batches(n_batches),
        contextlib.nullcontext(),
        dev,
        False
    )

    res = {
        "val_loss": loss,
        "perplexity": math.exp(loss),
        "tokens": n,
        "iter": meta.get("iter")
    }

    print(json.dumps(res, indent=2))

    return res


@torch.inference_mode()
def qa_eval(
    ckpt: str,
    sft_dir: str,
    n: int = 200,
    max_new: int = 64,
    device: str = "auto",
    show: int = 12
) -> Dict:

    model, tok, _ = load_model(
        ckpt,
        device=device
    )

    tokens = np.load(
        os.path.join(sft_dir, "val.tokens.npy"),
        mmap_mode="r"
    )

    mask = np.load(
        os.path.join(sft_dir, "val.mask.npy"),
        mmap_mode="r"
    )

    offsets = np.load(
        os.path.join(sft_dir, "val.offsets.npy")
    )

    total = len(offsets) - 1

    idxs = np.random.default_rng(0).permutation(total)[:n]

    em = 0
    f1 = 0
    ended = 0
    done = 0
    shown = 0

    for i in idxs:

        ids = [
            int(t)
            for t in tokens[offsets[i]:offsets[i + 1]]
        ]

        m = np.asarray(
            mask[offsets[i]:offsets[i + 1]]
        )

        first = int(np.argmax(m))

        try:
            j = ids.index(
                tok.end_id,
                first
            )
        except ValueError:
            continue

        prompt = ids[:first]

        ref = tok.decode(
            ids[first:j],
            skip_special=True
        )

        if len(prompt) + max_new > model.cfg.max_seq_len:
            continue

        pred_ids = list(
            stream_tokens(
                model,
                prompt,
                max_new_tokens=max_new,
                temperature=0.0,
                repetition_penalty=1.0,
                stop_ids=tok.stop_ids(),
                forbidden=tok.forbidden_in_answer(),
                valid_vocab=tok.vocab_size
            )
        )

        pred = tok.decode(
            pred_ids,
            skip_special=True
        ).strip()

        done += 1

        em += normalize(pred) == normalize(ref)
        f1 += f1_score(pred, ref)

        ended += len(pred_ids) < max_new

        if shown < show:

            q = tok.decode(
                prompt,
                skip_special=True
            )[-160:].replace("\n", " ")

            print(
                f"\nQ : …{q}\n"
                f"  attendu : {ref[:140]}\n"
                f"  obtenu  : {pred[:140]}"
            )

            shown += 1

    res = {
        "n": done,
        "exact_match": round(
            em / max(1, done),
            4
        ),
        "f1": round(
            f1 / max(1, done),
            4
        ),
        "ends_properly": round(
            ended / max(1, done),
            4
        )
    }

    print(
        "\n" + json.dumps(
            res,
            indent=2
        )
    )

    return res


# ============================================================
# QA OUVERT — AUCUN ATTENDU
# ============================================================

@torch.inference_mode()
def open_qa_eval(
    ckpt: str,
    n: int = 100,
    max_new: int = 80,
    device: str = "auto",
    temperature: float = 0.0,
    show_prompt_number: bool = True,
) -> None:

    model, tok, _ = load_model(
        ckpt,
        device=device
    )

    rng = np.random.default_rng(42)

    questions = list(UNSEEN_QA_PROMPTS)

    if n < len(questions):
        idxs = rng.choice(
            len(questions),
            size=n,
            replace=False
        )
        questions = [
            questions[int(i)]
            for i in idxs
        ]

    print("=" * 72)
    print("MiniLLM v2 — OPEN QA / QUESTIONS SANS ATTENDU")
    print(f"Questions : {len(questions)}")
    print("=" * 72)

    for i, question in enumerate(questions, 1):

        answer = chat_reply(
            model,
            tok,
            [
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=30,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram=3,
            seed=42 + i
        )

        if show_prompt_number:
            print(f"\n[{i}/{len(questions)}]")
        else:
            print()

        print(f"Q : {question}")
        print(f"A : {answer}")
        print("-" * 72)


@torch.inference_mode()
def persona_check(
    ckpt: str,
    persona_path: str = None,
    device: str = "auto",
    show: int = 10
) -> Dict:

    persona_path = (
        persona_path
        or os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "personnalite.jsonl"
        )
    )

    model, tok, _ = load_model(
        ckpt,
        device=device
    )

    rows = [
        json.loads(l)
        for l in open(
            persona_path,
            encoding="utf-8"
        )
        if l.strip()
    ]

    f1s = []
    exact = 0

    for i, r in enumerate(rows):

        pred = chat_reply(
            model,
            tok,
            [
                {
                    "role": "user",
                    "content": r["question"]
                }
            ],
            max_new_tokens=80,
            temperature=0.0,
            repetition_penalty=1.0
        )

        f = f1_score(
            pred,
            r["answer"]
        )

        f1s.append(f)

        exact += (
            normalize(pred)
            == normalize(r["answer"])
        )

        if i < show or f < 0.5:

            print(
                f"{'✓' if f >= 0.8 else '✗'} "
                f"{r['question']}\n"
                f"    attendu : {r['answer'][:100]}\n"
                f"    obtenu  : {pred[:100]}"
            )

    res = {
        "n": len(rows),
        "f1_moyen": round(
            sum(f1s) / max(1, len(f1s)),
            4
        ),
        "exact_match": round(
            exact / max(1, len(rows)),
            4
        )
    }

    print(
        "\n" + json.dumps(
            res,
            indent=2
        )
    )

    return res


def demo(
    ckpt: str,
    device: str = "auto",
    prompts: List[str] = None
) -> None:

    model, tok, _ = load_model(
        ckpt,
        device=device
    )

    for q in prompts or DEMO_PROMPTS:

        a = chat_reply(
            model,
            tok,
            [
                {
                    "role": "user",
                    "content": q
                }
            ],
            max_new_tokens=80,
            temperature=0.5,
            top_k=30,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram=3,
            seed=0
        )

        print(
            f"Toi      : {q}\n"
            f"MiniLLM  : {a}\n"
        )


def main():

    p = argparse.ArgumentParser(
        description="MiniLLM v2 — évaluation"
    )

    p.add_argument(
        "what",
        choices=[
            "perplexity",
            "qa",
            "openqa",
            "demo",
            "persona"
        ]
    )

    p.add_argument(
        "--persona",
        default=None,
        help="fichier de personnalité"
    )

    p.add_argument(
        "--ckpt",
        required=True
    )

    p.add_argument(
        "--data_dir",
        default="data/pretrain"
    )

    p.add_argument(
        "--sft_dir",
        default="data/sft"
    )

    p.add_argument(
        "--seq_len",
        type=int,
        default=512
    )

    p.add_argument(
        "--n",
        type=int,
        default=200
    )

    p.add_argument(
        "--max_new",
        type=int,
        default=80
    )

    p.add_argument(
        "--temperature",
        type=float,
        default=0.0
    )

    p.add_argument(
        "--device",
        default="auto"
    )

    a = p.parse_args()

    if a.what == "perplexity":

        perplexity(
            a.ckpt,
            a.data_dir,
            a.seq_len,
            device=a.device
        )

    elif a.what == "qa":

        qa_eval(
            a.ckpt,
            a.sft_dir,
            a.n,
            device=a.device
        )

    elif a.what == "openqa":

        open_qa_eval(
            a.ckpt,
            n=a.n,
            max_new=a.max_new,
            device=a.device,
            temperature=a.temperature
        )

    elif a.what == "persona":

        persona_check(
            a.ckpt,
            a.persona,
            device=a.device
        )

    else:

        demo(
            a.ckpt,
            a.device
        )


if __name__ == "__main__":
    main()