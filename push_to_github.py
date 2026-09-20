#!/usr/bin/env python3
"""
push_to_github.py — MiniLLM v2 : pousse le projet vers GitHub via l'API REST (git n'est pas nécessaire).
Pratique depuis un téléphone / Colab / Kaggle.

Usage (le token passe par une variable d'environnement : il n'apparaît ni dans l'historique ni dans le notebook) :
    export GITHUB_TOKEN=ghp_xxx              # ou  %env GITHUB_TOKEN=ghp_xxx  dans un notebook
    python push_to_github.py --repo minillm_v2 --dry_run          # voir ce qui serait poussé
    python push_to_github.py --repo minillm_v2 --delete_old       # pousse + supprime les anciens notebooks

Token conseillé : "fine-grained", limité à CE dépôt, permission « Contents : Read and write ».
Ne partage jamais un token dans un chat ; révoque-le s'il a fuité.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional

INCLUDE_EXT = {".py", ".md", ".txt", ".ipynb", ".json", ".jsonl"}
INCLUDE_NAMES = {".gitignore"}
SKIP_DIRS = {".git", "__pycache__", "data", "checkpoints", ".pytest_cache", "runs", "outputs"}
OLD_FILES = ["MiniLLM_v2_DevLab.ipynb", "MiniLLM_v2_Kaggle.ipynb"]      # remplacés par MiniLLM_v2.ipynb
MAX_BYTES = 5_000_000


def git_blob_sha(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


class GitHubAPI:
    BASE = "https://api.github.com"

    def __init__(self, token: str):
        self.headers = {
            "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, data: Optional[dict] = None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.BASE + path, data=body, headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GitHub API {e.code} : {e.read().decode()[:300]}") from None

    def file_info(self, owner: str, repo: str, path: str) -> Optional[dict]:
        try:
            return self.request("GET", f"/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}")
        except RuntimeError:
            return None


def collect_files(root: str) -> List[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            if os.path.splitext(fn)[1] in INCLUDE_EXT or fn in INCLUDE_NAMES:
                full = os.path.join(dirpath, fn)
                if os.path.getsize(full) <= MAX_BYTES:
                    out.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(out)


def push(token: str, repo: str, root: str, private: bool, dry_run: bool, delete_old: bool, message: str) -> None:
    api = GitHubAPI(token)
    owner = api.request("GET", "/user")["login"]
    print(f"Connecté : {owner}")
    try:
        api.request("GET", f"/repos/{owner}/{repo}")
    except RuntimeError:
        if dry_run:
            print(f"(le dépôt {owner}/{repo} n'existe pas encore : il serait créé)")
        else:
            api.request("POST", "/user/repos", {"name": repo, "private": private, "auto_init": False,
                                                "description": "MiniLLM v2 — petit LLM français from scratch"})
            print(f"Dépôt créé : {owner}/{repo}")

    pushed = unchanged = 0
    for path in collect_files(root):
        with open(os.path.join(root, path), "rb") as f:
            content = f.read()
        info = api.file_info(owner, repo, path)
        if info and info.get("sha") == git_blob_sha(content):
            unchanged += 1
            continue
        verb = "mise à jour" if info else "ajout"
        print(f"  {'[dry] ' if dry_run else ''}{verb:<12} {path} ({len(content) / 1024:.1f} Ko)")
        if not dry_run:
            payload = {"message": f"{message}: {path}", "content": base64.b64encode(content).decode()}
            if info:
                payload["sha"] = info["sha"]
            api.request("PUT", f"/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}", payload)
        pushed += 1

    if delete_old:
        for path in OLD_FILES:
            info = api.file_info(owner, repo, path)
            if info:
                print(f"  {'[dry] ' if dry_run else ''}suppression  {path}")
                if not dry_run:
                    api.request("DELETE", f"/repos/{owner}/{repo}/contents/{urllib.parse.quote(path)}",
                                {"message": f"{message}: remove {path}", "sha": info["sha"]})
    print(f"\n{pushed} fichier(s) {'à pousser' if dry_run else 'poussé(s)'}, {unchanged} inchangé(s).")
    print(f"https://github.com/{owner}/{repo}")


def main():
    p = argparse.ArgumentParser(description="MiniLLM v2 — push GitHub (API REST)")
    p.add_argument("--repo", default="minillm_v2")
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--delete_old", action="store_true", help="supprime les anciens notebooks DevLab/Kaggle")
    p.add_argument("--message", default="refactor v2")
    p.add_argument("--token", default=None, help="déconseillé : préfère la variable d'environnement GITHUB_TOKEN")
    a = p.parse_args()
    token = a.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("❌ Définis GITHUB_TOKEN (voir l'en-tête du fichier).")
    push(token, a.repo, a.dir, a.private, a.dry_run, a.delete_old, a.message)


if __name__ == "__main__":
    main()
