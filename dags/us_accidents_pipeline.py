from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# =========================
# CONFIG
# =========================
SPARK_CONTAINER = "spark"
JOBS_DIR = "/data/jobs"

# kalau kamu mau set env var tiap job (opsional)
COMMON_ENV = " ".join([
    # contoh:
    # "USE_CLASS_WEIGHT=1",
    # "USE_DOWNSAMPLE=0",
    # "CM_SAMPLE_N=200000",
])

def spark_submit(job_file: str) -> str:
    """
    Run spark-submit inside spark docker container.
    """
    # NOTE: path spark-submit di container kamu harus benar.
    # biasanya ada di /opt/spark/bin/spark-submit
    return f'docker exec {SPARK_CONTAINER} /opt/spark/bin/spark-submit {COMMON_ENV} {JOBS_DIR}/{job_file}'

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="us_accidents_bigdata_ml_dashboard",
    default_args=default_args,
    description="US Accidents pipeline: ETL -> Aggregates -> RF Train -> Scoring -> Graph -> Export -> Dashboard files",
    start_date=datetime(2026, 1, 1),
    schedule_interval=None,   # manual trigger dulu. nanti bisa pakai cron "0 2 * * *"
    catchup=False,
    max_active_runs=1,
    tags=["us-accidents", "spark", "ml", "graph", "streamlit"],
) as dag:

    t01_to_parquet = BashOperator(
        task_id="01_to_parquet",
        bash_command=spark_submit("01_to_parquet.py"),
    )

    t02_curate_features = BashOperator(
        task_id="02_curate_features",
        bash_command=spark_submit("02_curate_features.py"),
    )

    t03_aggregate_export = BashOperator(
        task_id="03_aggregate_export",
        bash_command=spark_submit("03_aggregate_export.py"),
    )

    t04_train_rf = BashOperator(
        task_id="04_train_rf",
        bash_command=spark_submit("04_train.py"),
    )

    t05_score_dashboard = BashOperator(
        task_id="05_score_and_dashboard",
        bash_command=spark_submit("05_score_and_dashboard.py"),
    )

    t06_graph_analytics = BashOperator(
        task_id="06_graph_analytics",
        bash_command=spark_submit("06_graph_analytics.py"),
    )

    t07_export_graph_csv = BashOperator(
        task_id="07_export_graph_to_csv",
        bash_command=spark_submit("07_export_graph_to_csv.py"),
    )

    # Pipeline order
    t01_to_parquet >> t02_curate_features >> t03_aggregate_export >> t04_train_rf >> t05_score_dashboard >> t06_graph_analytics >> t07_export_graph_csv
