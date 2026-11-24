"""
deploiement_prefect.py : planification des flows avec Prefect (cron).

Terminal 1 : prefect server start                (UI : http://localhost:4200)
Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
             python deploiement_prefect.py        (worker : exécute les flows)
Terminal 3 : prefect deployment ls
             prefect deployment run 'all/ml-pipeline-all'

Format cron : "minute heure jour mois jour_de_semaine"
    "0 2 * * *" -> tous les jours à 02:00 (heure de Tunis)
"""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import (
    all_flow,
    code_flow,
    evaluate_flow,
    git_flow,
    install_flow,
    push_flow,
    train_flow,
)

TIMEZONE = "Africa/Tunis"

if __name__ == "__main__":
    # serve() expose les flows, démarre un worker local et active le scheduler
    serve(
        # pipeline complet : tous les jours à 02:00 (heure de Tunis)
        all_flow.to_deployment(
            name="ml-pipeline-all",
            schedules=[CronSchedule(cron="0 2 * * *", timezone=TIMEZONE)],
            tags=["mlops", "full-pipeline"],
        ),
        # vérification du code + push : tous les jours à 01:00 (heure de Tunis)
        code_flow.to_deployment(
            name="ml-pipeline-code",
            schedules=[CronSchedule(cron="0 1 * * *", timezone=TIMEZONE)],
            tags=["mlops", "quality"],
        ),
        # lancement manuel uniquement (prefect deployment run ...)
        git_flow.to_deployment(name="ml-pipeline-git", tags=["mlops", "git"]),
        push_flow.to_deployment(name="ml-pipeline-push", tags=["mlops", "git"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["mlops", "setup"]),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["mlops", "training"]),
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate", tags=["mlops", "evaluation"]
        ),
    )
