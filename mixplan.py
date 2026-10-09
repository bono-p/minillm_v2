"""
mixplan.py — plan de mélange des données de pré-entraînement de miniLLM v2.3 (5 milliards de tokens).

Idée : planning WSD (warmup-stable-decay, déjà implémenté) + DEUX jeux de données.
  * phase STABLE (≈ 85 % des tokens) : mélange large (web, livres, science, administration, dialogues…) ;
  * phase DECAY  (≈ 15 % des tokens, la décroissance du learning rate) : texte moderne de bonne qualité + conversations
    multi-tours + questions-réponses. Pratique documentée pour les petits modèles (ex. MiniCPM met des données d'instruction
    dans la phase de décroissance). On change de données avec `train.py --stop_at <début de la décroissance>`.

Tout est une DONNÉE (dataclass) : on change les proportions en éditant DEFAULT_PLAN, sans toucher au code.

⚠️ Ce qui est VÉRIFIÉ et ce qui ne l'est pas :
  * vérifié (fiches Hugging Face) : licences et ordres de grandeur de Common Corpus (licences permissives), french_instruct (MIT,
    ≈ 274 000 conversations, ≈ 85 M tokens), french-instruction-dataset de vonewman (Apache-2.0, 108 991 conversations),
    Lucie-Training-Dataset et Claire (CC BY-NC-SA : exclus par défaut, car le modèle est publié sous MIT) ;
  * NON vérifié hors ligne : noms exacts des colonnes et organisation des fichiers de Common Corpus, taille réelle de chaque
    sous-ensemble français. `prepare_mix.py inspect` les affiche ; `est_available_tokens` n'est qu'une ESTIMATION qui sert à
    avertir quand une source devrait être répétée plus de ~3 fois.
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
    Source("web_wiki_existant", "existing_bin", "web_wiki", stable=int(1.45 * B), decay=int(0.22 * B),
           note="corpus v2.2 déjà tokenisé (Wikipédia FR + FineWeb-2 FR) : pas de retéléchargement ; la phase decay prend des blocs jamais utilisés en phase stable"),
    Source("livres_presse_fr", "hf_text", "livres", stable=int(1.40 * B), decay=int(0.10 * B), hf_path="PleIAs/common_corpus",
           filters={"collection": "OpenCulture", "language": "French"}, clean="livre", est_available_tokens=int(20 * B),
           note="livres et journaux du domaine public (OCR) : filtre qualité OCR ; style souvent ancien -> la phase decay (texte moderne) corrige. "
                "Une seule source (et non deux) : deux sources sur le même flux liraient les mêmes documents"),
    Source("openweb_fr", "hf_text", "openweb", stable=int(0.55 * B), decay=int(0.07 * B), hf_path="PleIAs/common_corpus",
           filters={"collection": "OpenWeb", "language": "French"}, est_available_tokens=int(5 * B),
           note="transcriptions YouTube-Commons et StackExchange en français (langue parlée, questions-réponses) ; la phase decay lit la suite du flux"),
    Source("science_fr", "hf_text", "science", stable=int(0.35 * B), decay=int(0.05 * B), hf_path="PleIAs/common_corpus",
           filters={"collection": "OpenScience", "language": "French"}, est_available_tokens=int(3 * B),
           note="thèses et articles scientifiques en français"),
    Source("admin_fr", "hf_text", "admin", stable=int(0.35 * B), hf_path="PleIAs/common_corpus",
           filters={"collection": "OpenGovernment", "language": "French"}, est_available_tokens=int(3 * B),
           note="textes administratifs et juridiques : registre formel, à ne pas laisser dominer"),
    Source("french_instruct", "hf_chat", "conversations", stable=int(0.10 * B), decay=int(0.20 * B),
           hf_path="MaziyarPanahi/french_instruct_sharegpt", text_field="conversations", est_available_tokens=int(85 * M), max_epochs=4.0,
           note="miroir ShareGPT de angeluriot/french_instruct (MIT), ≈ 85 M tokens, multi-tours ; une partie traduite via l'API ChatGPT (voir ses conditions d'utilisation)"),
    Source("french_instruction_vonewman", "hf_chat", "conversations", stable=int(0.05 * B), decay=int(0.10 * B),
           hf_path="vonewman/french-instruction-dataset", text_field="conversations", est_available_tokens=int(45 * M), max_epochs=4.0,
           note="Apache-2.0, 108 991 conversations (≈ 45 M tokens : estimation d'après la taille du fichier)"),
    Source("piaf_qa", "hf_qa", "qa", decay=int(0.01 * B), hf_path="etalab-ia/piaf", max_epochs=6.0, est_available_tokens=int(3 * M),
           note="questions-réponses sur Wikipédia (même source que le SFT) ; très petit, répété jusqu'à 6 fois"),
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
