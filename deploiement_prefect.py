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
                    cron="21 12 * * *",
                    timezone=TIMEZONE,
                )
            ],
            tags=["mlops", "full-pipeline"],
        )
    )
