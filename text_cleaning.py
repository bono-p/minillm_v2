"""
text_cleaning.py — nettoyage des corpus AVANT tokenisation.

Problème détecté à l'audit : le dump Wikipédia a perdu ses templates (dates, siècles, nombres), d'où des
phrases trouées dans les données d'entraînement :
    "né le  à Moulins (Allier) et mort le  à Châteaumeillant"   /   "des premières décennies du ."
Un modèle qui s'entraîne dessus apprend à produire des trous et des doubles espaces.
Ici on supprime les phrases défectueuses (et les paragraphes trop abîmés), on coupe les sections
"Notes et références / Liens externes..." et on retire les titres de section.
"""
from __future__ import annotations

import re
from typing import List, Optional

# Un motif = signe d'un template supprimé ou d'un formatage cassé.
_DEFECTS = [
    re.compile(r"\S {2,}\S"),                 # double espace au milieu d'une phrase ("né le  à")
    re.compile(r"\S\u00a0{2,}\S"),
    re.compile(r"\(\s*[,;.)]"),               # "( ," "( )" "(."
    re.compile(r"\[\s*[\],;.]"),              # "[ ]"
    re.compile(r"(?<![\s.])\s[.,](?![.\d])"), # espace AVANT un point/une virgule ("du .") : jamais légitime en français
    re.compile(r"(?:^|\s),(?:\s|$)"),          # virgule isolée (le " ; " est légitime en français)
    re.compile(r"\b(?:le|la|les|en|du|de|des|au|aux|à|vers)\s+[,;)]"),   # "en ," "du )" : mot-outil suivi d'un vide
    re.compile(r"\{\{|\}\}|\|\||==|<[a-z/][^>]*>"),  # restes de wikitexte / HTML
    re.compile(r"[,:;]\s*\)"),                   # "(en latin : )" "(né le , )" : parenthèse fermée sur du vide
    re.compile(r"\(\s*[:;]"),                    # "( : ...)" "( ; ...)"
    re.compile(r"\(\s*[-–—]+\s*\)"),           # "( - )"
    re.compile(r"«\s*»|“\s*”"),                  # guillemets vides
    re.compile(r"\b(?:le|la|les|en|du|de|des|au|aux|à|vers|depuis|entre|et|ou|par)\s+\)"),  # "(né le )", "(en )"
]

# Apostrophe d'élision suivie d'une espace ("L' archidiocèse", "d' Italie") : artefact de conversion des liens du dump.
# En français l'élision n'est JAMAIS suivie d'une espace -> on recolle.
_ELISION = re.compile(
    r"\b((?:[LlDdJjMmNnSsTtCc]|[Qq]u|[Jj]usqu|[Ll]orsqu|[Pp]uisqu|[Qq]uelqu|[Pp]resqu)['’]) +(?=[A-Za-zÀ-ÖØ-öø-ÿŒœ])"
)
_AUJOURDHUI = re.compile(r"\b([Aa]ujourd['’]) +(hui)\b")


def normalize_elisions(text: str) -> str:
    return _AUJOURDHUI.sub(r"\1\2", _ELISION.sub(r"\1", text))

# Sections de fin d'article : on coupe tout ce qui suit (listes de références, liens...).
_TAIL_SECTIONS = {
    "notes et références", "notes", "références", "référence", "liens externes", "voir aussi", "bibliographie",
    "articles connexes", "annexes", "sources", "notes et références", "publications", "filmographie",
    "discographie", "liens internes", "lien externe",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?…»])\s+(?=[A-ZÀ-ÖØ-Þ«\"“(\d])")
_TERMINAL = tuple(".!?…»)\"”")


def has_defect(sentence: str) -> bool:
    return any(p.search(sentence) for p in _DEFECTS)


def split_sentences(paragraph: str) -> List[str]:
    return [s for s in _SENT_SPLIT.split(paragraph.strip()) if s]


def is_heading(line: str) -> bool:
    """Titre de section : ligne courte sans ponctuation finale."""
    s = line.strip()
    return len(s) <= 90 and not s.endswith(_TERMINAL) and s.count(" ") <= 12


def clean_paragraph(paragraph: str, max_drop_frac: float = 0.4, stats: Optional[dict] = None) -> Optional[str]:
    sents = split_sentences(paragraph)
    if not sents:
        return None
    kept = [s for s in sents if not has_defect(s)]
    dropped = len(sents) - len(kept)
    if stats is not None:
        stats["sent_total"] = stats.get("sent_total", 0) + len(sents)
        stats["sent_dropped"] = stats.get("sent_dropped", 0) + dropped
    if not kept or dropped / len(sents) > max_drop_frac:
        return None
    return " ".join(kept)


def clean_wikipedia_text(text: str, min_chars: int = 300, stats: Optional[dict] = None) -> Optional[str]:
    """Nettoie UN article. Renvoie None si l'article est inutilisable."""
    paragraphs: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower().rstrip(" :") in _TAIL_SECTIONS:
            break                                       # fin d'article : références, liens externes...
        if is_heading(line):
            continue                                    # titres / intertitres / puces courtes
        cleaned = clean_paragraph(normalize_elisions(line), stats=stats)
        if cleaned and len(cleaned) >= 40:
            paragraphs.append(cleaned)
    doc = "\n\n".join(paragraphs)
    return doc if len(doc) >= min_chars else None


_SPAM_TERMS = (
    "cliquez ici", "en savoir plus", "livraison gratuite", "code promo", "sites de rencontre", "site de rencontre",
    "casino en ligne", "paris sportifs", "abonnez-vous", "newsletter", "politique de confidentialité", "cookies",
    "ajouter au panier", "acheter maintenant", "meilleur prix", "devis gratuit", "partager sur", "suivez-nous",
    "mentions légales", "tous droits réservés", "boutique en ligne", "offre spéciale",
)


def spam_score(text: str) -> int:
    """Nombre de termes de spam/boilerplate DISTINCTS présents (≥ 3 = page commerciale ou menu de site, pas un texte)."""
    low = text.lower()
    return sum(1 for t in _SPAM_TERMS if t in low)


def clean_web_text(text: str, min_chars: int = 400, max_chars: int = 20_000) -> Optional[str]:
    """Filtre léger pour du texte web déjà filtré (FineWeb-2)."""
    text = text.strip()
    if len(text) < min_chars:
        return None
    text = text[:max_chars]
    if spam_score(text) >= 3:
        return None
    letters = sum(c.isalpha() for c in text)
    if letters / max(1, len(text)) < 0.6:                # trop de chiffres / symboles
        return None
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(set(lines)) < 0.7 * len(lines):               # lignes répétées (menus, spam)
        return None
    return normalize_elisions("\n\n".join(lines))


def refilter_doc(text: str, src: str, min_chars: int = 300) -> Optional[str]:
    """
    Re-nettoie un document DÉJÀ présent dans corpus.jsonl (sans retélécharger) avec les règles actuelles.
    wiki : normalisation des élisions + suppression des phrases trouées ; autres sources : normalisation seule.
    """
    if src != "wiki":
        if src == "web" and spam_score(text) >= 3:
            return None
        return normalize_elisions(text)
    paragraphs = []
    for para in text.split("\n\n"):
        cleaned = clean_paragraph(normalize_elisions(para))
        if cleaned and len(cleaned) >= 40:
            paragraphs.append(cleaned)
    doc = "\n\n".join(paragraphs)
    return doc if len(doc) >= min_chars else None
