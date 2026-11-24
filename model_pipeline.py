"""
model_pipeline.py
-----------------
Pipeline ML modulaire : prédiction de la réadmission d'un patient diabétique.
Cible binaire : readmitted (0 = NO, 1 = '<30' ou '>30').

Contenu :
  1. Données      : load_data, clean_data, prepare_data
  2. Exploration  : explore_data (valeurs manquantes, cible, distributions,
                    outliers, corrélations)
  3. ACP          : run_pca
  4. Modèle       : train_model, tune_xgboost_grid, tune_xgboost_random,
                    tune_stacking_meta
  5. Évaluation   : evaluate_model, plot_confusion_matrix, plot_roc_curve,
                    plot_threshold_analysis, plot_feature_importance,
                    explain_shap
  6. Persistance  : save_model, load_model
"""

import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")  # pas d'affichage : les figures sont enregistrées en PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 100
REPORTS_DIR = "reports"

# Colonnes supprimées dans le notebook (> 39 % de valeurs manquantes)
COLS_TO_DROP = [
    "weight",
    "max_glu_serum",
    "A1Cresult",
    "payer_code",
    "medical_specialty",
]
ID_COLS = ["encounter_id", "patient_nbr"]

MEAN_COLS = ["time_in_hospital", "num_lab_procedures"]
MEDIAN_COLS = [
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
MODE_COLS = ["race", "diag_1", "diag_2", "diag_3"]
NUMERIC_COLS = MEAN_COLS + MEDIAN_COLS

# Meilleurs hyperparamètres trouvés par RandomizedSearchCV dans le notebook
BEST_XGB_PARAMS = {
    "subsample": 1.0,
    "reg_lambda": 1,
    "reg_alpha": 10,
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.1,
    "gamma": 0,
    "colsample_bytree": 0.8,
}


# --------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------
def _save_fig(name, output_dir=REPORTS_DIR):
    """Enregistre la figure courante dans output_dir et la ferme."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  figure enregistrée : {path}")
    return path


def _make_xgb(**params):
    """XGBClassifier avec la configuration commune du notebook."""
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", random_state=42, **params
    )


# --------------------------------------------------------------------------
# 1. Données
# --------------------------------------------------------------------------
def load_data(path):
    """Charge le CSV et remplace '?' par NaN."""
    return pd.read_csv(path).replace("?", np.nan)


def clean_data(df):
    """Supprime les colonnes inutiles, impute, binarise la cible, encode.

    Returns:
        tuple: (DataFrame encodé, dict des LabelEncoder)
    """
    df = df.drop(columns=COLS_TO_DROP, errors="ignore").copy()

    for col in MEAN_COLS:
        df[col] = df[col].fillna(df[col].mean())
    for col in MEDIAN_COLS:
        df[col] = df[col].fillna(df[col].median())
    for col in MODE_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.dropna(subset=["readmitted"])
    df["readmitted"] = df["readmitted"].map({"NO": 0, "<30": 1, ">30": 1}).astype(int)

    label_encoders = {}
    # "string" : colonnes texte de pandas 3 (StringDtype), "object" : pandas 2
    for col in df.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    return df, label_encoders


def prepare_data(path, test_size=0.2, random_state=RANDOM_STATE):
    """Charge, nettoie, sépare en train/test et standardise.

    Le scaler est ajusté sur le train uniquement (pas de fuite de données).

    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    df = load_data(path)
    if "readmitted" not in df.columns:
        raise ValueError("La colonne cible 'readmitted' est absente du fichier.")

    df, _ = clean_data(df)
    df = df.drop(columns=ID_COLS, errors="ignore")

    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns, index=X_test.index
    )
    return X_train, X_test, y_train, y_test, scaler


# --------------------------------------------------------------------------
# 2. Exploration (cellules 4 à 31 du notebook)
# --------------------------------------------------------------------------
def explore_data(path, mapping_path=None, output_dir=REPORTS_DIR):
    """Analyse exploratoire : dimensions, valeurs manquantes, cible,
    distributions, outliers (IQR) et matrice de corrélation.
    Les graphiques sont enregistrés dans output_dir.
    """
    raw = pd.read_csv(path)
    print("Dimensions du jeu de données :", raw.shape)
    if mapping_path and os.path.exists(mapping_path):
        print("Dimensions de ids_mapping    :", pd.read_csv(mapping_path).shape)

    raw.info()

    # Valeurs manquantes codées '?'
    qm = (raw == "?").sum()
    print("\nValeurs '?' par colonne :\n", qm[qm > 0])

    df = raw.replace("?", np.nan)
    missing = df.isnull().mean() * 100
    missing[missing > 0].plot(kind="bar", figsize=(10, 5))
    plt.title("Pourcentage des valeurs manquantes par variable")
    plt.ylabel("Pourcentage (%)")
    plt.xticks(rotation=45, ha="right")
    _save_fig("01_valeurs_manquantes.png", output_dir)

    # Après suppression des colonnes trop vides
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")
    print("\nDimensions après suppression des colonnes :", df.shape)
    missing = df.isnull().mean() * 100
    if (missing > 0).any():
        missing[missing > 0].plot(kind="bar", figsize=(10, 5))
        plt.title("Valeurs manquantes (après suppression des colonnes)")
        plt.ylabel("Pourcentage (%)")
        plt.xticks(rotation=45, ha="right")
        _save_fig("02_valeurs_manquantes_apres.png", output_dir)

    # Distribution de la cible
    print("\nDistribution de readmitted :")
    print(df["readmitted"].value_counts())
    print(df["readmitted"].value_counts(normalize=True).round(3))
    plt.figure(figsize=(6, 4))
    sns.countplot(x="readmitted", data=df, order=["NO", ">30", "<30"])
    plt.title("Distribution de la variable cible")
    _save_fig("03_distribution_cible.png", output_dir)

    # Histogrammes + boxplots des variables numériques
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    fig, axes = plt.subplots(len(cols), 2, figsize=(12, 3.2 * len(cols)))
    for i, col in enumerate(cols):
        axes[i, 0].hist(df[col].dropna(), bins=20)
        axes[i, 0].set_title(f"Distribution de {col}")
        axes[i, 1].boxplot(df[col].dropna())
        axes[i, 1].set_title(f"Boxplot de {col}")
    _save_fig("04_distributions_numeriques.png", output_dir)

    # Données nettoyées : outliers, histogrammes globaux, corrélations
    clean, _ = clean_data(pd.read_csv(path).replace("?", np.nan))

    print("\nValeurs aberrantes (méthode IQR) :")
    for col in cols:
        q1, q3 = clean[col].quantile(0.25), clean[col].quantile(0.75)
        iqr = q3 - q1
        n_out = ((clean[col] < q1 - 1.5 * iqr) | (clean[col] > q3 + 1.5 * iqr)).sum()
        print(f"  {col} : {n_out} valeurs aberrantes")

    clean.hist(figsize=(20, 15))
    _save_fig("05_histogrammes_globaux.png", output_dir)

    normalized = clean.drop(columns=ID_COLS, errors="ignore").copy()
    feat = [c for c in normalized.columns if c != "readmitted"]
    normalized[feat] = StandardScaler().fit_transform(normalized[feat])
    plt.figure(figsize=(22, 16))
    sns.heatmap(
        normalized.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 6}
    )
    plt.title("Matrice de corrélation - Diabetes Dataset")
    _save_fig("06_matrice_correlation.png", output_dir)


# --------------------------------------------------------------------------
# 3. ACP (cellules 35 à 41 du notebook)
# --------------------------------------------------------------------------
def run_pca(X_train, y_train, output_dir=REPORTS_DIR):
    """ACP : tableau des valeurs propres, scree plot (critère de Kaiser),
    carte des individus, cercle de corrélation et biplot.

    Returns:
        tuple: (objet PCA ajusté, DataFrame des valeurs propres)
    """
    data_numeric = X_train.select_dtypes(include=[np.number]).dropna()
    data_scaled = StandardScaler().fit_transform(data_numeric)
    print("Dimensions des données pour l'ACP :", data_scaled.shape)

    pca = PCA()
    comps = pca.fit_transform(data_scaled)

    eig = pd.DataFrame(
        {
            "Dimension": [f"Dim{i + 1}" for i in range(len(pca.explained_variance_))],
            "Variance expliquée": pca.explained_variance_,
            "% variance expliquée": np.round(pca.explained_variance_ratio_ * 100, 2),
            "% cum. var. expliquée": np.round(
                np.cumsum(pca.explained_variance_ratio_) * 100, 2
            ),
        }
    )
    print(eig.to_string(index=False))

    # Scree plot + critère de Kaiser
    ev = pca.explained_variance_
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ev) + 1), ev, marker="o", color="b")
    plt.axhline(y=1, color="r", linestyle="--", label="Critère de Kaiser")
    plt.title("Valeurs propres des composantes principales")
    plt.xlabel("Composante principale")
    plt.ylabel("Valeur propre")
    plt.legend()
    plt.grid(True)
    _save_fig("07_acp_scree_plot.png", output_dir)

    labels = pd.Series(y_train.values).map({0: "Non réadmis", 1: "Réadmis"})
    pca_df = pd.DataFrame(comps[:, :2], columns=["PC1", "PC2"])
    pca_df["readmitted"] = labels.values
    xl = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f} %)"
    yl = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f} %)"

    # Carte des individus
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="PC1", y="PC2", data=pca_df, hue="readmitted", alpha=0.5)
    plt.axhline(0, color="grey", linewidth=1)
    plt.axvline(0, color="grey", linewidth=1)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title("Carte des individus selon la réadmission")
    _save_fig("08_acp_individus.png", output_dir)

    # Cercle de corrélation
    names = data_numeric.columns
    pcs = pca.components_ * np.sqrt(pca.explained_variance_[:, np.newaxis])
    plt.figure(figsize=(10, 10))
    plt.axhline(0, color="grey", linestyle="--")
    plt.axvline(0, color="grey", linestyle="--")
    for i, name in enumerate(names):
        plt.arrow(
            0,
            0,
            pcs[0, i],
            pcs[1, i],
            color="b",
            alpha=0.5,
            head_width=0.03,
            length_includes_head=True,
        )
        plt.text(pcs[0, i] * 1.1, pcs[1, i] * 1.1, name, fontsize=8, ha="center")
    plt.gca().add_artist(
        plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--")
    )
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title("Cercle de corrélation des variables numériques")
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.grid()
    plt.axis("equal")
    _save_fig("09_acp_cercle_correlation.png", output_dir)

    # Biplot
    scale = 5
    plt.figure(figsize=(14, 10))
    sns.scatterplot(x="PC1", y="PC2", data=pca_df, hue="readmitted", alpha=0.3, s=15)
    plt.axhline(0, color="grey", linewidth=1)
    plt.axvline(0, color="grey", linewidth=1)
    for i, name in enumerate(names):
        plt.arrow(
            0,
            0,
            pcs[0, i] * scale,
            pcs[1, i] * scale,
            color="red",
            alpha=0.7,
            head_width=0.12,
            length_includes_head=True,
        )
        plt.text(
            pcs[0, i] * scale * 1.12,
            pcs[1, i] * scale * 1.12,
            name,
            color="red",
            fontsize=8,
            ha="center",
            va="center",
        )
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title("Biplot : individus et variables numériques")
    plt.grid(alpha=0.2)
    _save_fig("10_acp_biplot.png", output_dir)

    return pca, eig


# --------------------------------------------------------------------------
# 4. Modèle (cellules 44 à 51 du notebook)
# --------------------------------------------------------------------------
def _build_stacking(final_estimator=None):
    """StackingClassifier XGBoost + RandomForest + régression logistique."""
    return StackingClassifier(
        estimators=[
            (
                "xgb",
                _make_xgb(
                    colsample_bytree=0.9,
                    learning_rate=0.1,
                    max_depth=6,
                    n_estimators=200,
                    subsample=0.8,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
                ),
            ),
        ],
        final_estimator=final_estimator
        or LogisticRegression(random_state=42, max_iter=1000),
        cv=3,
        n_jobs=-1,
    )


def train_model(X_train, y_train, model_type="xgboost", params=None):
    """Entraîne un modèle de classification.

    Args:
        model_type (str): 'xgboost' ou 'stacking'
        params (dict): hyperparamètres XGBoost (défaut : BEST_XGB_PARAMS)
    """
    if model_type == "xgboost":
        model = _make_xgb(**(params or BEST_XGB_PARAMS))
    elif model_type == "stacking":
        model = _build_stacking()
    else:
        raise ValueError("model_type doit être 'xgboost' ou 'stacking'.")
    model.fit(X_train, y_train)
    return model


def tune_xgboost_grid(X_train, y_train):
    """GridSearchCV (scoring = accuracy, cv = 3) - cellule 45.

    Returns:
        tuple: (meilleur modèle, meilleurs paramètres, meilleur score CV)
    """
    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [5, 6, 8],
        "n_estimators": [150, 200],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
    }
    search = GridSearchCV(
        _make_xgb(),
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    start = time.time()
    search.fit(X_train, y_train)
    print(f"Optimisation terminée en {time.time() - start:.2f} s")
    print("Meilleurs paramètres :", search.best_params_)
    print(f"Meilleure accuracy (CV) : {search.best_score_ * 100:.2f}%")
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_xgboost_random(X_train, y_train, n_iter=10):
    """RandomizedSearchCV (scoring = roc_auc, cv = 3) - cellule 46."""
    param_dist = {
        "learning_rate": [0.1, 0.01],
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.5, 1],
        "reg_alpha": [0, 1, 10],
        "reg_lambda": [0, 1, 10],
    }
    search = RandomizedSearchCV(
        XGBClassifier(objective="binary:logistic", random_state=42),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=42,
    )
    start = time.time()
    search.fit(X_train, y_train)
    print(f"RandomizedSearchCV terminé en {time.time() - start:.2f} s")
    print("Meilleurs paramètres :", search.best_params_)
    print(f"Meilleur ROC AUC (CV) : {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


def tune_stacking_meta(X_train, y_train):
    """GridSearchCV sur le méta-modèle (régression logistique) du stacking - cellule 51.
    Attention : cette étape est longue (plusieurs minutes).
    """
    param_grid = {
        "final_estimator__C": [0.01, 0.1, 1.0, 10.0],
        "final_estimator__penalty": ["l2"],
    }
    search = GridSearchCV(
        _build_stacking(),
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )
    start = time.time()
    search.fit(X_train, y_train)
    print(f"Optimisation terminée en {time.time() - start:.2f} s")
    print("Meilleurs paramètres du méta-modèle :", search.best_params_)
    return search.best_estimator_, search.best_params_, search.best_score_


# --------------------------------------------------------------------------
# 5. Évaluation et interprétation
# --------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Évalue le modèle (rapport, matrice de confusion, accuracy, ROC AUC).

    Args:
        threshold (float): seuil de décision (0.4 favorise le rappel)
    Returns:
        dict: accuracy, roc_auc, confusion_matrix
    """
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    print(f"=== Évaluation (seuil = {threshold}) ===")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion :\n", metrics["confusion_matrix"])
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"ROC AUC  : {metrics['roc_auc']:.4f}")
    return metrics


def plot_confusion_matrix(model, X_test, y_test, threshold=0.5, output_dir=REPORTS_DIR):
    """Matrice de confusion normalisée + taux (spécificité, rappel, FP, FN)."""
    proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, (proba >= threshold).astype(int))
    cmn = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=["Non réadmis (0)", "Réadmis (1)"],
        yticklabels=["Non réadmis (0)", "Réadmis (1)"],
        annot_kws={"size": 12},
    )
    plt.title(f"Matrice de confusion normalisée (seuil = {threshold})")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    _save_fig(f"11_matrice_confusion_seuil_{threshold}.png", output_dir)

    print(f"Spécificité (vrais négatifs)   : {cmn[0, 0] * 100:.2f}%")
    print(f"Faux positifs (erreur type I)  : {cmn[0, 1] * 100:.2f}%")
    print(f"Faux négatifs (erreur type II) : {cmn[1, 0] * 100:.2f}%")
    print(f"Rappel (vrais positifs)        : {cmn[1, 1] * 100:.2f}%")
    return cmn


def plot_roc_curve(model, X_test, y_test, output_dir=REPORTS_DIR):
    """Courbe ROC."""
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="blue", lw=2, label=f"Courbe ROC (AUC = {auc(fpr, tpr):.2f})"
    )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig("12_courbe_roc.png", output_dir)


def plot_threshold_analysis(model, X_test, y_test, output_dir=REPORTS_DIR):
    """Précision et rappel en fonction du seuil de décision - cellule 49."""
    precisions, recalls, thresholds = precision_recall_curve(
        y_test, model.predict_proba(X_test)[:, 1]
    )
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Précision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Rappel (sensibilité)", linewidth=2)
    plt.axvline(x=0.5, color="gray", linestyle=":", label="Seuil par défaut (0.5)")
    plt.xlabel("Seuil de décision")
    plt.ylabel("Score")
    plt.title("Précision et rappel selon le seuil de décision")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig("13_precision_rappel_seuil.png", output_dir)


def plot_feature_importance(model, feature_names, top_n=15, output_dir=REPORTS_DIR):
    """Importance des variables (modèles disposant de feature_importances_,
    ex. XGBoost)."""
    if not hasattr(model, "feature_importances_"):
        print("Ce modèle n'expose pas feature_importances_ (ex. stacking) : ignoré.")
        return None
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(
        ascending=False
    )

    plt.figure(figsize=(10, 6))
    imp.head(top_n)[::-1].plot(kind="barh")
    plt.title(f"Top {top_n} des variables les plus importantes")
    _save_fig("14_importance_top.png", output_dir)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=imp.values, y=imp.index, orient="h")
    plt.title("Importance des caractéristiques")
    plt.xlabel("Importance")
    _save_fig("15_importance_complete.png", output_dir)
    return imp


def explain_shap(model, X_sample, max_samples=2000, output_dir=REPORTS_DIR):
    """Explication SHAP globale (nécessite `pip install shap` et un modèle à arbres)."""
    try:
        import shap
    except ImportError:
        print("SHAP n'est pas installé : pip install shap")
        return None
    try:
        sample = X_sample.sample(min(max_samples, len(X_sample)), random_state=42)
        shap_values = shap.TreeExplainer(model).shap_values(sample)
        shap.summary_plot(shap_values, sample, show=False)
        return _save_fig("16_shap_summary.png", output_dir)
    except Exception as exc:  # modèle non supporté (ex. stacking)
        print("Note concernant SHAP :", exc)
        return None


# --------------------------------------------------------------------------
# 6. Persistance
# --------------------------------------------------------------------------
def save_model(model, path="model.joblib"):
    """Sauvegarde un modèle (ou tout objet, ex. scaler) avec joblib."""
    joblib.dump(model, path)
    print(f"Sauvegardé : {path}")


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    return joblib.load(path)
