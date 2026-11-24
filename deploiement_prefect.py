
"""
deploiement_prefect.py : planification du pipeline avec Prefect.

Terminal 1 : prefect server start
             UI : http://localhost:4200

Terminal 2 : prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
             python deploiement_prefect.py

Terminal 3 : prefect deployment ls
             prefect deployment run 'all/ml-pipeline-all'

Format cron : "minute heure jour mois jour_de_semaine"
"0 2 * * *" -> tous les jours à 02:00 (heure de Tunis)
"""

from prefect import serve
from prefect.client.schemas.schedules import CronSchedule

from pipeline_prefect import all_flow

TIMEZONE = "Africa/Tunis"

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            schedules=[
                CronSchedule(
                cron="5 12 * * *",
                    timezone=TIMEZONE,
                )
            ],
            tags=["mlops", "full-pipeline"],
        )
    )
