"""
deploiement_prefect.py : planification des flows avec Prefect.

Terminal 1 : prefect server start                (UI : http://localhost:4200)
Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
             python deploiement_prefect.py        (worker : exécute les flows)
Terminal 3 : prefect deployment ls
             prefect deployment run 'all/ml-pipeline-all'
"""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    train_flow,
)

if __name__ == "__main__":
    serve(
        # pipeline complet tous les jours à 02:00
        all_flow.to_deployment(
            name="ml-pipeline-all", cron="0 2 * * *", tags=["full-pipeline", "mlops"]
        ),
        # vérification du code tous les jours à 01:00
        code_flow.to_deployment(
            name="ml-pipeline-code", cron="0 1 * * *", tags=["quality", "mlops"]
        ),
        # lancement manuel uniquement
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]
        ),
    )
