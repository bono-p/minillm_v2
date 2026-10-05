"""
make_banner.py — régénère assets/banner.png avec de VRAIES réponses du modèle (panneau de droite).

Le fond, le titre et les pastilles sont conservés de la bannière précédente ; seules les lignes du dialogue sont redessinées
(même police DejaVu Sans, mêmes couleurs). Les réponses affichées sont passées en paramètre : ne jamais y mettre un texte
qui n'a pas été produit par le modèle publié.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

BG = (27, 27, 41)
TEAL, LIGHT, YELLOW, GREY = (78, 205, 196), (201, 199, 214), (242, 196, 109), (110, 108, 126)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_I = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"


def _wrap(draw, text: str, font, max_w: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split(" "):
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def render(src: str, dst: str, exchanges: List[Tuple[str, str, str]], size: int = 28, x0: int = 1214, max_w: int = 600) -> None:
    """exchanges : [(question, réponse, commentaire éventuel)] ; au plus 2 échanges pour tenir dans le panneau."""
    im = Image.open(src).convert("RGB")
    d = ImageDraw.Draw(im)
    d.rectangle((1195, 150, 1855, 485), fill=BG)
    f, fi = ImageFont.truetype(FONT, size), ImageFont.truetype(FONT_I, size)
    y = 172
    for q, a, note in exchanges:
        d.text((x0, y), ">", font=f, fill=TEAL)
        x = x0 + d.textlength("> ", font=f)
        for part, col in (("chat_reply(", LIGHT), (f'"{q}"', YELLOW), (")", LIGHT)):
            d.text((x, y), part, font=f, fill=col)
            x += d.textlength(part, font=f)
        y += 48
        lines = _wrap(d, f'"{a}"', f, max_w)
        for k, ln in enumerate(lines):
            d.text((x0, y), ln, font=f, fill=YELLOW)
            if k == len(lines) - 1 and note:
                d.text((x0 + d.textlength(ln, font=f) + 36, y), note, font=fi, fill=GREY)
            y += 38
        y += 38
    im.save(dst)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    render(src, dst, [("Qui es-tu ?", "Je suis MiniLLM, un petit modèle de langage qui essaie de répondre à tes questions en français.", ""),
                      ("Combien font 2 plus 3 ?", "2 plus 3 font 10.", "// faux, assumé")])
