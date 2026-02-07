# US Accidents Big Data Analytics Project

This project is a big data analytics pipeline for analyzing US accidents dataset using Apache Spark. The pipeline performs ETL, feature engineering, data aggregation, machine learning model training (Random Forest), scoring, graph analytics, and data export for dashboard.

## Dataset

The dataset used is from Kaggle:
https://www.kaggle.com/datasets/mlgodsiddharth/usa-accidents-dataset49-states-subset-of

The dataset contains accident data from 49 US states from 2016 to 2023.

## Project Structure

- `dags/` - Airflow DAG for pipeline orchestration
- `jobs/` - Spark jobs for processing:
  - `01_to_parquet.py` - Convert CSV to Parquet format
  - `02_curate_features.py` - Feature engineering
  - `03_aggregate_export.py` - Data aggregation and export
  - `04_train.py` - Random Forest model training
  - `05_score_and_dashboard.py` - Scoring and dashboard preparation
  - `06_graph_analytics.py` - Graph analytics
  - `07_export_graph_to_csv.py` - Export graph to CSV
- `data_lake/` - Data lake for storing processed data, curated data, ML models, and serving layer
- `dataset/` - Input CSV dataset
- `docker-compose.yaml` - Docker configuration for Spark container

## Technologies

- Apache Spark 3.5.7
- Apache Airflow
- Python 3
- Docker

## How to Run

1. Make sure Docker is installed
2. Run `docker-compose up -d` to start Spark container
3. Setup Airflow and trigger DAG `us_accidents_bigdata_ml_dashboard`
4. The pipeline will process data sequentially from ETL to result export
