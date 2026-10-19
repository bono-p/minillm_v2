"""
mixplan.py — plan de mélange des données de pré-entraînement de miniLLM v2.3 (5 milliards de tokens).

Idée : planning WSD (warmup-stable-decay, déjà implémenté) + DEUX jeux de données.
  * phase STABLE (≈ 85 % des tokens) : mélange large (web, livres, science, administration, dialogues…) ;
  * phase DECAY  (≈ 15 % des tokens, la décroissance du learning rate) : texte moderne de bonne qualité + conversations
    multi-tours + questions-réponses. Pratique documentée pour les petits modèles (ex. MiniCPM met des données d'instruction
    dans la phase de décroissance). On change de données avec `train.py --stop_at <début de la décroissance>`.

Tout est une DONNÉE (dataclass) : on change les proportions en éditant DEFAULT_PLAN, sans toucher au code.

⚠️ Ce qui est VÉRIFIÉ et ce qui ne l'est pas :
  * vérifié (fiches Hugging Face, puis confirmé en lisant des lignes réelles le 2026-10-10) : licences et ordres de grandeur de
    Common Corpus (licences permissives) ; champ `collection` = sous-collection précise (ex. « French Open Data », « Wikidata »),
    champ `open_type` = l'un des 6 grands groupes (« Open Culture », « Open Government », « Open Science », « Open Web »,
    « Open Source », « Semantic data ») — c'est CE champ qu'on filtre, pas `collection` ; champ `language` = « French » pour les
    lignes en français (confirmé sur un exemple « Open Government ») ; french_instruct (MIT, ≈ 274 000 conversations, ≈ 85 M
    tokens, colonne `conversations` au format ShareGPT), french-instruction-dataset de vonewman (Apache-2.0, 108 991
    conversations, même format), PIAF (colonnes `context`/`question`/`answers.text`) ;
  * encore NON vérifié hors ligne : que « Open Culture », « Open Science » et « Open Web » s'écrivent exactement ainsi (seuls
    « Open Government » et « Semantic data » ont été vus sur un exemple réel) — si un filtre ne donne aucune ligne, `prepare_mix.py
    build` s'arrête avant de consommer du quota et affiche les valeurs d'`open_type` réellement rencontrées ; taille réelle de
    chaque sous-ensemble français. `est_available_tokens` n'est qu'une ESTIMATION qui sert à avertir quand une source devrait
    être répétée plus de ~3 fois.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

B = 1_000_000_000
M = 1_000_000


@dataclass
class Source:
    name: str
    kind: str                       # "existing_bin" | "hf_text" | "hf_chat" | "hf_qa"
    category: str                   # web_wiki | livres | openweb | science | admin | presse | conversations | qa
    stable: int = 0                 # tokens visés en phase STABLE
    decay: int = 0                  # tokens visés en phase DECAY
    hf_path: str = ""
    hf_config: str = ""
    hf_data_files: str = ""         # motif de fichiers (ex. "OpenCulture/*") pour ne pas parcourir tout un gros dépôt
    filters: Dict[str, str] = field(default_factory=dict)   # ex. {"language": "French", "collection": "OpenCulture"}
    text_field: str = "text"
    license: str = "permissive"     # "permissive" | "nc" (usage non commercial : exclu par défaut) | "unknown"
    clean: str = ""                 # "" | "livre" | "presse" : nettoyage OCR
    max_epochs: float = 3.0         # répétitions maximales de la source
    est_available_tokens: int = 0   # ESTIMATION (non vérifiée) ; 0 = inconnue
    note: str = ""

    @property
    def total(self) -> int:
        return self.stable + self.decay

    @property
    def est_epochs(self) -> float:
        return self.total / self.est_available_tokens if self.est_available_tokens else 0.0


# ── plan par défaut : 5,00 milliards de tokens = 4,25 (stable) + 0,75 (decay) ─────────────────────────────────────────────────────
DEFAULT_PLAN: List[Source] = [
    Source("wikipedia_fr", "hf_text", "web_wiki", stable=int(1.00 * B), decay=int(0.20 * B), hf_path="wikimedia/wikipedia", hf_config="20231101.fr",
           clean="wiki", est_available_tokens=int(1.3 * B), license="permissive",
           note="Wikipédia FR (CC BY-SA), mêmes nettoyages que la v2.2 ; relevé le 11/10/2026 : 0,95 Md obtenus sans aucune répétition -> augmenté"),
    Source("fineweb2_fr", "hf_text", "web_wiki", stable=int(2.94 * B), decay=int(0.30 * B), hf_path="HuggingFaceFW/fineweb-2", hf_config="fra_Latn",
           clean="web", est_available_tokens=int(50 * B),
           note="web français filtré (FineWeb-2, ODC-By) ; reçoit l'essentiel du budget libéré par Common Corpus (source quasi inépuisable, 0 répétition relevée)"),
    # Les 4 sources Common Corpus ci-dessous sont PLAFONNÉES à ce qui a été réellement obtenu le 11/10/2026 (en butant sur
    # max_epochs) : le français représenté sous ces `open_type` est beaucoup plus rare dans ce dépôt que les ~20 Md estimés au
    # départ (quelques millions à quelques dizaines de millions de tokens seulement). Quota = acquis -> un nouveau `build` les
    # saute sans rescanner. Le manque a été reporté sur fineweb2_fr, qui n'a pas cette limite.
    Source("livres_presse_fr", "hf_text", "livres", stable=74_060_000, decay=18_515_000, hf_path="PleIAs/common_corpus",
           filters={"open_type": "Open Culture", "language": "French"}, clean="livre", est_available_tokens=93_000_000,
           note="livres et journaux du domaine public (OCR) en français : source petite dans Common Corpus (~93 M tokens), plafonnée à l'acquis"),
    Source("openweb_fr", "hf_text", "openweb", stable=793_000, decay=198_000, hf_path="PleIAs/common_corpus",
           filters={"open_type": "Open Web", "language": "French"}, est_available_tokens=1_000_000,
           note="quasiment aucun contenu en français sous cet `open_type` dans Common Corpus (~1 M tokens) : gardé pour mémoire, part négligeable"),
    Source("science_fr", "hf_text", "science", stable=7_901_000, decay=1_975_000, hf_path="PleIAs/common_corpus",
           filters={"open_type": "Open Science", "language": "French"}, est_available_tokens=9_900_000,
           note="thèses et articles scientifiques en français : source petite (~10 M tokens), plafonnée à l'acquis"),
    Source("admin_fr", "hf_text", "admin", stable=8_718_000, hf_path="PleIAs/common_corpus",
           filters={"open_type": "Open Government", "language": "French"}, est_available_tokens=8_800_000,
           note="textes administratifs et juridiques en français : source petite (~9 M tokens), plafonnée à l'acquis"),
    Source("french_instruct", "hf_chat", "conversations", stable=int(0.10 * B), decay=int(0.20 * B),
           hf_path="MaziyarPanahi/french_instruct_sharegpt", text_field="conversations", est_available_tokens=int(85 * M), max_epochs=6.0,
           note="miroir ShareGPT de angeluriot/french_instruct (MIT), ≈ 85 M tokens, multi-tours ; une partie traduite via l'API ChatGPT (voir ses conditions d'utilisation)"),
    Source("french_instruction_vonewman", "hf_chat", "conversations", stable=int(0.05 * B), decay=int(0.10 * B),
           hf_path="vonewman/french-instruction-dataset", text_field="conversations", est_available_tokens=int(45 * M), max_epochs=7.0,
           note="Apache-2.0, 108 991 conversations (≈ 45 M tokens : estimation d'après la taille du fichier)"),
    Source("piaf_qa", "hf_qa", "qa", decay=2_000_000, hf_path="etalab-ia/piaf", max_epochs=6.0, est_available_tokens=700_000,
           note="questions-réponses sur Wikipédia (même source que le SFT) ; très petit (~700 k tokens/époque), répété jusqu'à 6 fois. "
                "Corrige un bug du 11/10/2026 : prepare_mix.py appelait _piaf_window() avec le mauvais signature -> 100% des lignes rejetées"),
]

# ── candidates HORS du plan par défaut (en attente d'une décision, n'affectent jamais `build` sans --only) ─────────────
# Conversations Reddit FR (demande du 12/10/2026). FrancophonIA/LELU (556 621 conversations, 1,58 M tours de parole)
# vient d'un export public BigQuery du contenu Reddit, repris via un dataset Kaggle : je n'ai pas trouvé de mention
# claire de licence autorisant explicitement cette réutilisation (les conditions d'utilisation de Reddit encadrent la
# redistribution en masse de contenu utilisateur), ni confirmé le nom réel de la colonne qui contiendrait des
# conversations groupées (à vérifier avec `prepare_mix.py tree --repo FrancophonIA/LELU` et `inspect`). `license="unknown"`
# fait que `validate()` la signale et que `build` refuse de la traiter tant qu'elle n'est pas explicitement ajoutée au
# plan ET que la licence n'est pas tranchée (en repassant `license="permissive"` ici, en connaissance de cause).
EXTRA_CANDIDATES: List[Source] = [
    Source("reddit_fr", "hf_chat", "conversations", decay=int(0.05 * B), hf_path="FrancophonIA/LELU",
           text_field="conversations", est_available_tokens=int(60 * M), max_epochs=5.0, license="unknown",
           note="conversations Reddit FR nettoyées ; licence ET nom de colonne à vérifier avant tout usage"),
]

# Sources interdites par défaut (licence non commerciale) : listées pour mémoire, jamais dans le plan par défaut.
EXCLUDED_NC = {
    "OpenLLM-France/Lucie-Training-Dataset": "CC BY-NC-SA 4.0 (fiche Hugging Face)",
    "OpenLLM-France/Claire-Dialogue-French-0.1": "CC BY-NC-SA 4.0 (fiche Hugging Face) : dialogues oraux, utile mais incompatible avec une publication MIT",
}


def phase_total(plan: List[Source], phase: str) -> int:
    return sum(getattr(s, phase) for s in plan)


def category_shares(plan: List[Source]) -> Dict[str, float]:
    tot = sum(s.total for s in plan) or 1
    out: Dict[str, float] = {}
    for s in plan:
        out[s.category] = out.get(s.category, 0.0) + s.total / tot
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def validate(plan: List[Source], allow_nc: bool = False, budget: int = 0, tol: float = 0.02) -> List[str]:
    """Liste de problèmes (vide = plan acceptable)."""
    issues: List[str] = []
    names = [s.name for s in plan]
    if len(set(names)) != len(names):
        issues.append("noms de sources en double")
    for s in plan:
        if s.license == "nc" and not allow_nc:
            issues.append(f"{s.name} : licence non commerciale — incompatible avec une publication MIT (passe allow_nc=True pour une variante de recherche non publiée)")
        if s.license == "unknown":
            issues.append(f"{s.name} : licence inconnue — à vérifier avant d'entraîner")
        if s.total <= 0:
            issues.append(f"{s.name} : aucun token demandé")
        if s.kind == "hf_text" and not s.hf_path:
            issues.append(f"{s.name} : hf_path manquant")
        if s.est_epochs > s.max_epochs:
            issues.append(f"{s.name} : ≈ {s.est_epochs:.1f} répétitions prévues (> {s.max_epochs:g}) — baisse le quota ou trouve plus de données")
    if budget:
        got = sum(s.total for s in plan)
        if abs(got - budget) > tol * budget:
            issues.append(f"budget total {got / B:.2f} Md ≠ visé {budget / B:.2f} Md")
    return issues


def storage_bytes(plan: List[Source], bytes_per_token: int = 2) -> int:
    """Octets écrits sur disque (fichiers train des deux phases ; la validation est négligeable)."""
    return sum(s.total for s in plan) * bytes_per_token


def summary(plan: List[Source]) -> str:
    lines = [f"{'source':<30} {'catégorie':<14} {'stable':>8} {'decay':>8} {'total':>8} {'part':>6} {'époques≈':>9}  licence", "-" * 100]
    tot = sum(s.total for s in plan) or 1
    for s in plan:
        ep = f"{s.est_epochs:.1f}" if s.est_available_tokens else "-"
        lines.append(f"{s.name:<30} {s.category:<14} {s.stable / B:8.2f} {s.decay / B:8.2f} {s.total / B:8.2f} {s.total / tot * 100:5.1f}% {ep:>9}  {s.license}")
    lines.append("-" * 100)
    lines.append(f"{'TOTAL (milliards de tokens)':<45} {phase_total(plan, 'stable') / B:8.2f} {phase_total(plan, 'decay') / B:8.2f} {tot / B:8.2f}")
    lines.append("")
    lines.append("Par catégorie : " + " · ".join(f"{c} {p * 100:.0f} %" for c, p in category_shares(plan).items()))
    lines.append(f"Disque : {storage_bytes(plan) / 1e9:.1f} Go (uint16) pour les deux phases + ≈ 0,1 Go de validation.")
    return "\n".join(lines)


def scaled(plan: List[Source], budget: int) -> List[Source]:
    """Même proportions, autre budget total (ex. 3 Md pour un essai, 10 Md plus tard)."""
    import dataclasses
    k = budget / max(1, sum(s.total for s in plan))
    return [dataclasses.replace(s, stable=int(s.stable * k), decay=int(s.decay * k)) for s in plan]


if __name__ == "__main__":
    print(summary(DEFAULT_PLAN))
    for p in validate(DEFAULT_PLAN, budget=5 * B):
        print("⚠️ ", p)
