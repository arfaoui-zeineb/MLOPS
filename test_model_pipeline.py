"""Tests du projet : tests unitaires + tests de sécurité et de qualité.

Ce fichier unique regroupe :
  PARTIE 1 - Tests unitaires de model_pipeline.py (données, modèle, figures,
             persistance). Ils utilisent un petit jeu de données synthétique
             (même structure que diabetic_data.csv) pour rester rapides.
  PARTIE 2 - Tests de sécurité (bandit) et de qualité du code (flake8 lint).

Remarques données lors de la validation et prise en compte :
  1. « Sécurité avec bandit et lint flake8 »
     -> PARTIE 2 : bandit et flake8 sont exécutés comme des tests pytest sur
        model_pipeline.py, main.py, pipeline_prefect.py et sur ce fichier.
        En cas d'échec, le fichier, la ligne et la règle en cause sont affichés.
        (B101 « assert » est ignoré uniquement ici : pytest repose sur assert.)
  2. « L'affichage des tests »
     -> les tests sont lancés avec « pytest -v » : une ligne par test avec son
        nom explicite et son résultat (PASSED / FAILED), puis les 3 tests les
        plus lents (--durations=3).
  3. « Comment faire la connexion avec git »
     -> tâche « Récupération du projet (git pull) » dans pipeline_prefect.py,
        première étape du flow all (et flow git seul). Le dépôt distant est
        relié une fois avec : git remote add origin <url> ; git push -u origin main

Lancement (-s affiche les messages « Test N completed ») :
    pytest -v -s test_model_pipeline.py
    python pipeline_prefect.py --flow code
"""

import json

# subprocess : lance flake8 et bandit avec des arguments fixes, sans shell
import subprocess  # nosec B404
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import model_pipeline as mp

ROOT = Path(__file__).resolve().parent

# Fichiers du projet contrôlés par les tests de qualité et de sécurité
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
TEST_FILE = "test_model_pipeline.py"

N_ROWS = 80
FAST_XGB = {"n_estimators": 10, "max_depth": 2}


def _synthetic_df(n=N_ROWS, seed=0):
    """Construit un DataFrame qui imite diabetic_data.csv (avec des '?')."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "encounter_id": np.arange(n),
            "patient_nbr": np.arange(n) + 1000,
            "race": rng.choice(["Caucasian", "AfricanAmerican", "?"], n),
            "gender": rng.choice(["Female", "Male"], n),
            "age": rng.choice(["[50-60)", "[60-70)", "[70-80)"], n),
            "weight": "?",
            "time_in_hospital": rng.integers(1, 14, n).astype(float),
            "payer_code": rng.choice(["MC", "?"], n),
            "medical_specialty": rng.choice(["Cardiology", "?"], n),
            "num_lab_procedures": rng.integers(1, 100, n).astype(float),
            "num_procedures": rng.integers(0, 6, n).astype(float),
            "num_medications": rng.integers(1, 40, n).astype(float),
            "number_outpatient": rng.integers(0, 5, n).astype(float),
            "number_emergency": rng.integers(0, 3, n).astype(float),
            "number_inpatient": rng.integers(0, 4, n).astype(float),
            "diag_1": rng.choice(["250.83", "276", "428", "?"], n),
            "diag_2": rng.choice(["250.01", "427", "?"], n),
            "diag_3": rng.choice(["255", "403", "?"], n),
            "number_diagnoses": rng.integers(1, 10, n).astype(float),
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "insulin": rng.choice(["No", "Up", "Down", "Steady"], n),
            "readmitted": np.tile(["NO", "<30", ">30", "NO"], n // 4),
        }
    )
    # quelques valeurs numériques manquantes à imputer
    df.loc[:3, "time_in_hospital"] = np.nan
    df.loc[:3, "num_medications"] = np.nan
    return df


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "diabetic_data.csv"
    _synthetic_df().to_csv(path, index=False)
    return str(path)


@pytest.fixture
def split_data(csv_path):
    return mp.prepare_data(csv_path)


@pytest.fixture
def trained_model(split_data):
    X_train, _, y_train, _, _ = split_data
    return mp.train_model(X_train, y_train, params=FAST_XGB)


# ==========================================================================
# PARTIE 1 : TESTS UNITAIRES
# ==========================================================================


# ---------------------------------------------------------------- données
def test_load_data_remplace_les_points_d_interrogation(csv_path):
    df = mp.load_data(csv_path)
    assert len(df) == N_ROWS
    assert not (df == "?").any().any()
    assert df["race"].isna().any()
    print("\n  Test 1 completed : load_data remplace les '?' par NaN")


def test_clean_data_supprime_impute_et_encode():
    df, encoders = mp.clean_data(_synthetic_df().replace("?", np.nan))

    for col in mp.COLS_TO_DROP:
        assert col not in df.columns
    assert df.isna().sum().sum() == 0
    assert set(df["readmitted"].unique()) <= {0, 1}
    assert df.select_dtypes(include=["object", "string"]).empty
    assert {"gender", "age", "race"} <= set(encoders)
    print("\n  Test 2 completed : clean_data supprime, impute et encode")


def test_clean_data_binarise_la_cible():
    df, _ = mp.clean_data(_synthetic_df().replace("?", np.nan))
    attendu = _synthetic_df()["readmitted"].map({"NO": 0, "<30": 1, ">30": 1})
    assert (df["readmitted"].values == attendu.values).all()
    print("\n  Test 3 completed : clean_data binarise la cible readmitted")


def test_prepare_data_dimensions_et_standardisation(split_data):
    X_train, X_test, y_train, y_test, scaler = split_data

    assert len(X_train) + len(X_test) == N_ROWS
    assert len(X_test) == pytest.approx(0.2 * N_ROWS, abs=1)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert list(X_train.columns) == list(X_test.columns)
    for col in mp.ID_COLS + ["readmitted"]:
        assert col not in X_train.columns
    # le scaler est ajusté sur le train : moyenne ~0 sur le train
    assert np.allclose(X_train.mean().values, 0, atol=1e-6)
    assert hasattr(scaler, "mean_")
    print("\n  Test 4 completed : prepare_data : dimensions et standardisation")


def test_prepare_data_sans_cible_leve_une_erreur(tmp_path):
    path = tmp_path / "sans_cible.csv"
    _synthetic_df().drop(columns=["readmitted"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="readmitted"):
        mp.prepare_data(str(path))
    print("\n  Test 5 completed : prepare_data sans cible lève une erreur")


# ---------------------------------------------------------------- modèle
def test_train_model_xgboost_predit_des_classes_binaires(trained_model, split_data):
    _, X_test, _, _, _ = split_data
    preds = trained_model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)) <= {0, 1}
    print("\n  Test 6 completed : train_model XGBoost prédit 0 ou 1")


def test_train_model_type_invalide(split_data):
    X_train, _, y_train, _, _ = split_data
    with pytest.raises(ValueError):
        mp.train_model(X_train, y_train, model_type="inconnu")
    print("\n  Test 7 completed : train_model refuse un type invalide")


def test_evaluate_model_renvoie_des_metriques_valides(trained_model, split_data):
    _, X_test, _, y_test, _ = split_data
    metrics = mp.evaluate_model(trained_model, X_test, y_test, threshold=0.4)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert metrics["confusion_matrix"].sum() == len(y_test)
    print("\n  Test 8 completed : evaluate_model renvoie des métriques valides")


# ---------------------------------------------------------------- figures
def test_plot_feature_importance_enregistre_les_figures(
    trained_model, split_data, tmp_path
):
    X_train = split_data[0]
    imp = mp.plot_feature_importance(
        trained_model, X_train.columns, output_dir=str(tmp_path)
    )
    assert len(imp) == X_train.shape[1]
    assert (tmp_path / "14_importance_top.png").exists()
    assert (tmp_path / "15_importance_complete.png").exists()
    print("\n  Test 9 completed : importance des variables enregistrée")


def test_plot_feature_importance_ignore_les_modeles_sans_importance(tmp_path):
    assert mp.plot_feature_importance(object(), [], output_dir=str(tmp_path)) is None
    print("\n  Test 10 completed : modèle sans importance ignoré")


def test_run_pca_variance_cumulee_atteint_100(split_data, tmp_path):
    X_train, _, y_train, _, _ = split_data
    pca, eig = mp.run_pca(X_train, y_train, output_dir=str(tmp_path))

    assert eig["% cum. var. expliquée"].iloc[-1] == pytest.approx(100, abs=0.1)
    assert (tmp_path / "07_acp_scree_plot.png").exists()
    print("\n  Test 11 completed : ACP : variance cumulée = 100 %")


# ---------------------------------------------------------------- persistance
def test_save_et_load_model_redonnent_les_memes_predictions(
    trained_model, split_data, tmp_path
):
    _, X_test, _, _, _ = split_data
    path = str(tmp_path / "model.joblib")

    mp.save_model(trained_model, path)
    reloaded = mp.load_model(path)

    np.testing.assert_array_equal(
        trained_model.predict_proba(X_test), reloaded.predict_proba(X_test)
    )
    print("\n  Test 12 completed : sauvegarde / chargement du modèle")


# ==========================================================================
# PARTIE 2 : TESTS DE SÉCURITÉ (bandit) ET DE QUALITÉ (flake8)
# Remarque de validation n°1 : « sécurité avec bandit et lint flake8 »
# ==========================================================================
def _outil(*args):
    """Lance un outil Python (flake8, bandit) sur le projet et renvoie le résultat."""
    return subprocess.run(  # nosec B603
        [sys.executable, "-m", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _bandit(*args):
    """Lance bandit en sortie JSON et renvoie la liste des problèmes trouvés."""
    result = _outil("bandit", "-f", "json", *args)
    return json.loads(result.stdout)["results"]


def test_flake8_aucune_erreur_de_style():
    result = _outil("flake8", *PROJECT_FILES, TEST_FILE)
    assert result.returncode == 0, f"Erreurs flake8 :\n{result.stdout}"
    print("\n  Test 13 completed : flake8 : aucune erreur de style")


def test_bandit_aucune_faille_dans_le_code():
    problemes = _bandit(*PROJECT_FILES)
    assert problemes == [], "Failles bandit :\n" + "\n".join(
        f"{p['filename']}:{p['line_number']} [{p['test_id']}] {p['issue_text']}"
        for p in problemes
    )
    print("\n  Test 14 completed : bandit : aucune faille dans le code")


def test_bandit_aucune_faille_dans_les_tests():
    # B101 (assert) ignoré : pytest repose sur assert
    problemes = _bandit("-s", "B101", TEST_FILE)
    assert problemes == [], "Failles bandit :\n" + "\n".join(
        f"{p['filename']}:{p['line_number']} [{p['test_id']}] {p['issue_text']}"
        for p in problemes
    )
    print("\n  Test 15 completed : bandit : aucune faille dans les tests")
