"""
pipeline_prefect.py : orchestration du pipeline ML avec Prefect.

Flows disponibles :
    python pipeline_prefect.py --flow git        # récupération du projet (git pull)
    python pipeline_prefect.py --flow push       # commit + push des modifications
    python pipeline_prefect.py --flow install    # dépendances
    python pipeline_prefect.py --flow code       # black, flake8, bandit, tests, push
    python pipeline_prefect.py --flow train      # préparation + entraînement
    python pipeline_prefect.py --flow evaluate   # chargement + évaluation
    python pipeline_prefect.py --flow all        # pull + code + modèle + push

Le push automatique n'a lieu qu'après la réussite des tests.
"""

import argparse
import shutil

# subprocess : commandes fixes lancées sans shell, donc sans injection possible
import subprocess  # nosec B404
import sys
from datetime import datetime
from pathlib import Path

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from main import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetic_data.csv"
MODEL_PATH = BASE_DIR / "model.joblib"
SCALER_PATH = BASE_DIR / "scaler.joblib"

# Les étapes de qualité et de sécurité ne ciblent que les fichiers du projet
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
TEST_FILES = ["test_model_pipeline.py"]


def _run(*args):
    """Lance un outil Python du venv courant dans le dossier du projet."""
    cmd = [sys.executable, "-m", *args]
    print("$", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, cwd=BASE_DIR, check=True)  # nosec B603


def _git(*args, capture=False):
    """Lance une commande git (chemin absolu de l'exécutable, sans shell)."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git n'est pas installé (sudo apt install git).")
    print("$ git", " ".join(args), flush=True)
    result = subprocess.run(  # nosec B603
        [git, *args], cwd=BASE_DIR, check=True, capture_output=capture, text=True
    )
    return result.stdout


# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------
def _git_pret():
    """Vrai si le projet est un dépôt git relié à un remote 'origin'."""
    if not (BASE_DIR / ".git").exists():
        print("Pas de dépôt git local : faire 'git init' puis 'git remote add'.")
        return False
    if "origin" not in _git("remote", capture=True).split():
        print("Aucun remote 'origin' : faire 'git remote add origin <url>'.")
        return False
    return True


@task(name="Récupération du projet (git pull)")
def pull_project():
    """Met à jour le projet depuis le dépôt distant (remote 'origin')."""
    if _git_pret():
        _git("pull", "--ff-only")


@task(name="Envoi du projet (git push)")
def push_project(message=None):
    """Commit automatique des modifications puis push vers 'origin'.

    Appelée après les tests : seul un code validé est envoyé.
    """
    if not _git_pret():
        return
    _git("add", "-A")
    if _git("status", "--porcelain", capture=True).strip():
        date = datetime.now().strftime("%Y-%m-%d %H:%M")
        _git("commit", "-m", message or f"Mise à jour automatique ({date})")
    else:
        print("Aucune modification à committer.")
    _git("push", "origin", "HEAD")


@task(name="Installer les dépendances")
def install_dependencies():
    _run("pip", "install", "-q", "-r", "requirements.txt")


@task(name="Formatage du code")
def format_code():
    _run("black", *PROJECT_FILES, *TEST_FILES)


@task(name="Qualité du code")
def check_quality():
    _run("flake8", *PROJECT_FILES, *TEST_FILES)


@task(name="Sécurité du code")
def check_security():
    _run("bandit", "-q", *PROJECT_FILES)
    # B101 (assert) ignoré dans les tests : pytest repose sur assert
    _run("bandit", "-q", "-s", "B101", *TEST_FILES)


@task(name="Tests unitaires")
def run_tests():
    # -v : une ligne par test (nom + PASSED/FAILED) ; -s : messages « completed »
    _run("pytest", "-v", "-s", "--color=yes", "--durations=3", *TEST_FILES)


@task(name="Préparation des données", cache_policy=NO_CACHE)
def prepare(data_path=DATA_PATH):
    X_train, X_test, y_train, y_test, scaler = prepare_data(str(data_path))
    print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


@task(name="Entraînement du modèle", cache_policy=NO_CACHE)
def train(X_train, y_train, model_type="xgboost"):
    return train_model(X_train, y_train, model_type=model_type)


@task(name="Sauvegarde du modèle", cache_policy=NO_CACHE)
def save(model, scaler):
    save_model(model, str(MODEL_PATH))
    save_model(scaler, str(SCALER_PATH))


@task(name="Chargement du modèle")
def load():
    return load_model(str(MODEL_PATH))


@task(name="Évaluation du modèle", cache_policy=NO_CACHE)
def evaluate(model, X_test, y_test, threshold=0.5):
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    return {"accuracy": metrics["accuracy"], "roc_auc": metrics["roc_auc"]}


@task(name="Prédiction", cache_policy=NO_CACHE)
def predict(model, X, n=5):
    preds = model.predict(X.head(n))
    print(f"Prédictions sur {n} patients :", preds.tolist())
    return preds.tolist()


# --------------------------------------------------------------------------
# Flows
# --------------------------------------------------------------------------
@flow(name="git")
def git_flow():
    pull_project()


@flow(name="push")
def push_flow():
    push_project()


@flow(name="install")
def install_flow():
    install_dependencies()


@flow(name="code")
def code_flow(push=True):
    format_code()
    check_quality()
    check_security()
    run_tests()
    # si une étape échoue, Prefect s'arrête avant : rien n'est envoyé
    if push:
        push_project()


@flow(name="train")
def train_flow(model_type="xgboost"):
    X_train, _, y_train, _, scaler = prepare()
    model = train(X_train, y_train, model_type)
    save(model, scaler)


@flow(name="evaluate")
def evaluate_flow(threshold=0.5):
    _, X_test, _, y_test, _ = prepare()
    model = load()
    metrics = evaluate(model, X_test, y_test, threshold)
    predict(model, X_test)
    return metrics


@flow(name="all")
def all_flow(model_type="xgboost", threshold=0.5):
    pull_project()
    install_dependencies()
    code_flow(push=False)
    X_train, X_test, y_train, y_test, scaler = prepare()
    model = train(X_train, y_train, model_type)
    save(model, scaler)
    metrics = evaluate(model, X_test, y_test, threshold)
    predict(model, X_test)
    push_project()
    return metrics


FLOWS = {
    "git": git_flow,
    "push": push_flow,
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "all": all_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect")
    parser.add_argument("--flow", choices=FLOWS, default="all")
    FLOWS[parser.parse_args().flow]()
