#!/usr/bin/env python3
# push_to_github.py — MiniLLM v2
#
# Crée le repo GitHub et pousse tous les fichiers via l'API GitHub.
# Ne nécessite pas git installé — utilise uniquement l'API REST.
#
# Usage :
#   python push_to_github.py --token TON_TOKEN
#   python push_to_github.py --token TON_TOKEN --repo minillm_v2 --private
#
# Obtenir un token :
#   GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
#   Permissions requises : repo (toutes les cases)

import os
import sys
import json
import base64
import argparse
import urllib.request
import urllib.error

# ══════════════════════════════════════════════════════════════════════════════
#  API GitHub helper
# ══════════════════════════════════════════════════════════════════════════════

class GitHubAPI:
    BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.token   = token
        self.headers = {
            "Authorization":        f"Bearer {token}",
            "Accept":               "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type":         "application/json",
        }

    def _request(self, method: str, path: str, data: dict = None):
        url     = f"{self.BASE}{path}"
        body    = json.dumps(data).encode() if data else None
        req     = urllib.request.Request(url, data=body, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            raise RuntimeError(f"GitHub API {e.code} : {body}")

    def get_user(self) -> dict:
        return self._request("GET", "/user")

    def create_repo(self, name: str, description: str, private: bool) -> dict:
        return self._request("POST", "/user/repos", {
            "name":         name,
            "description":  description,
            "private":      private,
            "auto_init":    False,
            "has_issues":   True,
            "has_wiki":     False,
        })

    def get_repo(self, owner: str, repo: str) -> dict | None:
        try:
            return self._request("GET", f"/repos/{owner}/{repo}")
        except RuntimeError:
            return None

    def put_file(
        self,
        owner:   str,
        repo:    str,
        path:    str,
        content: bytes,
        message: str,
        sha:     str = None,
    ) -> dict:
        data = {
            "message": message,
            "content": base64.b64encode(content).decode(),
        }
        if sha:
            data["sha"] = sha
        return self._request("PUT", f"/repos/{owner}/{repo}/contents/{path}", data)

    def get_file_sha(self, owner: str, repo: str, path: str) -> str | None:
        """Récupère le SHA d'un fichier existant (nécessaire pour le mettre à jour)."""
        try:
            r = self._request("GET", f"/repos/{owner}/{repo}/contents/{path}")
            return r.get("sha")
        except RuntimeError:
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  Push principal
# ══════════════════════════════════════════════════════════════════════════════

FILES = [
    "config.py",
    "model.py",
    "train.py",
    "generate.py",
    "prepare_data.py",
    "inspect_model.py",
    "requirements.txt",
    "README.md",
    "MiniLLM_v2_DevLab.ipynb",
]


def push_to_github(
    token:       str,
    repo_name:   str  = "minillm_v2",
    description: str  = "🧠 MiniLLM v2 — Transformer LLM optimisé (RoPE+RMSNorm+SwiGLU+GQA), scalable 15M→1B",
    private:     bool = False,
    source_dir:  str  = ".",
):
    api = GitHubAPI(token)

    # ── Vérifier le token ────────────────────────────────────────────────
    print("Vérification du token GitHub...")
    user = api.get_user()
    username = user["login"]
    print(f"  ✓ Connecté en tant que : {username}")

    # ── Créer ou récupérer le repo ────────────────────────────────────────
    existing = api.get_repo(username, repo_name)
    if existing:
        print(f"  ℹ  Repo déjà existant : {existing['html_url']}")
        repo_url = existing["html_url"]
    else:
        print(f"  Création du repo '{repo_name}'...")
        new_repo = api.create_repo(repo_name, description, private)
        repo_url = new_repo["html_url"]
        print(f"  ✓ Repo créé : {repo_url}")

    # ── Pousser chaque fichier ────────────────────────────────────────────
    print(f"\nPush des fichiers vers {username}/{repo_name} ...")
    print(f"{'─'*50}")

    success = 0
    skipped = 0
    errors  = 0

    for filename in FILES:
        filepath = os.path.join(source_dir, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠  {filename:35s} → FICHIER ABSENT, ignoré")
            skipped += 1
            continue

        with open(filepath, "rb") as f:
            content = f.read()

        size_kb = len(content) / 1024

        # Vérifier si le fichier existe déjà sur GitHub
        sha = api.get_file_sha(username, repo_name, filename)
        msg = f"feat: update {filename}" if sha else f"feat: add {filename}"

        try:
            api.put_file(username, repo_name, filename, content, msg, sha)
            action = "mise à jour" if sha else "ajouté"
            print(f"  ✓  {filename:35s} ({size_kb:.1f} KB)  → {action}")
            success += 1
        except RuntimeError as e:
            print(f"  ✗  {filename:35s} → ERREUR : {e}")
            errors += 1

    # ── Résumé ────────────────────────────────────────────────────────────
    print(f"{'─'*50}")
    print(f"  ✓ {success} fichiers poussés")
    if skipped:
        print(f"  ⚠  {skipped} fichiers absents ignorés")
    if errors:
        print(f"  ✗ {errors} erreurs")
    print()
    print(f"🎉 Repo disponible : {repo_url}")
    print()
    print(f"📓 Ouvrir dans Colab :")
    nb_url = f"https://colab.research.google.com/github/{username}/{repo_name}/blob/main/MiniLLM_v2_DevLab.ipynb"
    print(f"   {nb_url}")
    print()
    print(f"   (ou clique sur le badge Open in Colab dans le README)")

    return repo_url


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MiniLLM v2 — Push automatique vers GitHub via API REST"
    )
    p.add_argument("--token",   required=True,
                   help="GitHub Personal Access Token (scope: repo)")
    p.add_argument("--repo",    default="minillm_v2",
                   help="Nom du repo (défaut: minillm_v2)")
    p.add_argument("--private", action="store_true",
                   help="Rendre le repo privé")
    p.add_argument("--dir",     default=".",
                   help="Dossier contenant les fichiers du projet")
    args = p.parse_args()

    if not args.token or args.token == "TON_TOKEN":
        print("❌ Fournis ton Personal Access Token avec --token")
        print()
        print("Comment l'obtenir :")
        print("  1. Va sur github.com → Settings → Developer settings")
        print("  2. Personal access tokens → Tokens (classic) → Generate new token")
        print("  3. Coche 'repo' (toutes les sous-cases)")
        print("  4. Copie le token et relance :")
        print(f"     python push_to_github.py --token ghp_XXXX...")
        sys.exit(1)

    push_to_github(
        token     = args.token,
        repo_name = args.repo,
        private   = args.private,
        source_dir= args.dir,
    )
