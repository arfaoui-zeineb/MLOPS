"""
main.py : exécution des étapes du pipeline via la ligne de commande.

Exemples :
    python main.py --explore                       # EDA + graphiques (reports/)
    python main.py --pca                           # ACP
    python main.py --tune random --train-tuned     # recherche d'hyperparamètres
    python main.py --train                         # entraîne (XGBoost) et sauvegarde
    python main.py --train --model stacking
    python main.py --evaluate --threshold 0.4
    python main.py --report                        # confusion, ROC, seuil, importance
    python main.py --shap                          # explication SHAP
    python main.py --train --evaluate --report     # enchaîner
"""

import argparse

from model_pipeline import (
    evaluate_model,
    explain_shap,
    explore_data,
    load_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
    plot_threshold_analysis,
    prepare_data,
    run_pca,
    save_model,
    train_model,
    tune_stacking_meta,
    tune_xgboost_grid,
    tune_xgboost_random,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline ML - Réadmission des patients diabétiques"
    )
    p.add_argument("--data", default="diabetic_data.csv", help="Chemin du CSV")
    p.add_argument(
        "--mapping", default="IDS_mapping.csv", help="Chemin de IDS_mapping.csv"
    )
    p.add_argument("--model-path", default="model.joblib", help="Fichier du modèle")
    p.add_argument("--scaler-path", default="scaler.joblib", help="Fichier du scaler")
    p.add_argument("--reports-dir", default="reports", help="Dossier des figures")
    p.add_argument("--model", choices=["xgboost", "stacking"], default="xgboost")
    p.add_argument("--threshold", type=float, default=0.5, help="Seuil de décision")

    p.add_argument("--explore", action="store_true", help="Analyse exploratoire (EDA)")
    p.add_argument(
        "--prepare", action="store_true", help="Préparer les données seulement"
    )
    p.add_argument("--pca", action="store_true", help="ACP")
    p.add_argument(
        "--tune",
        choices=["grid", "random", "stacking"],
        help="Recherche d'hyperparamètres (le meilleur modèle est sauvegardé)",
    )
    p.add_argument("--train", action="store_true", help="Entraîner et sauvegarder")
    p.add_argument(
        "--evaluate", action="store_true", help="Évaluer le modèle sauvegardé"
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Graphiques : confusion, ROC, seuil, importance",
    )
    p.add_argument("--shap", action="store_true", help="Explication SHAP")
    return p.parse_args()


def main():
    args = parse_args()
    actions = [
        args.explore,
        args.prepare,
        args.pca,
        args.tune,
        args.train,
        args.evaluate,
        args.report,
        args.shap,
    ]
    if not any(actions):
        print("Aucune action choisie. Voir : python main.py --help")
        return

    if args.explore:
        print("== Exploration des données ==")
        explore_data(args.data, args.mapping, args.reports_dir)

    needs_data = any(
        [
            args.prepare,
            args.pca,
            args.tune,
            args.train,
            args.evaluate,
            args.report,
            args.shap,
        ]
    )
    if needs_data:
        print("== Préparation des données ==")
        X_train, X_test, y_train, y_test, scaler = prepare_data(args.data)
        print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")

    if args.pca:
        print("== ACP ==")
        run_pca(X_train, y_train, args.reports_dir)

    if args.tune:
        print(f"== Optimisation ({args.tune}) ==")
        tuner = {
            "grid": tune_xgboost_grid,
            "random": tune_xgboost_random,
            "stacking": tune_stacking_meta,
        }[args.tune]
        best_model, _, _ = tuner(X_train, y_train)
        save_model(best_model, args.model_path)
        save_model(scaler, args.scaler_path)

    if args.train:
        print(f"== Entraînement ({args.model}) ==")
        model = train_model(X_train, y_train, model_type=args.model)
        save_model(model, args.model_path)
        save_model(scaler, args.scaler_path)

    if args.evaluate or args.report or args.shap:
        model = load_model(args.model_path)

        if args.evaluate:
            evaluate_model(model, X_test, y_test, threshold=args.threshold)

        if args.report:
            print("== Rapport graphique ==")
            plot_confusion_matrix(
                model, X_test, y_test, args.threshold, args.reports_dir
            )
            plot_roc_curve(model, X_test, y_test, args.reports_dir)
            plot_threshold_analysis(model, X_test, y_test, args.reports_dir)
            plot_feature_importance(model, X_train.columns, output_dir=args.reports_dir)

        if args.shap:
            print("== SHAP ==")
            explain_shap(model, X_train, output_dir=args.reports_dir)


if __name__ == "__main__":
    main()
