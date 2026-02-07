# 03_aggregate_export.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ✅ FIX: harus sama dengan OUT_PARQUET di script 2
INPUT_PATH = "/data/data_lake/curated/us_accidents_features"

# output kamu (boleh tetap seperti ini)
OUT_ANALYTICS = "/data/data_lake/analytics/us_accidents_summary_csv"
OUT_ML = "/data/data_lake/ml/us_accidents_features_parquet"

spark = (
    SparkSession.builder
    .appName("03_aggregate_export")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)

df = spark.read.parquet(INPUT_PATH)

# ------------ (A) SUMMARY UNTUK POWER BI / EXCEL ------------
summary = (
    df.groupBy("year", "State", "Severity")
      .agg(
          F.count("*").alias("accident_count"),
          F.avg("Distance_mi").alias("avg_distance_mi"),
          F.avg("Temperature_F").alias("avg_temp_f"),
          F.avg("Humidity_pct").alias("avg_humidity_pct"),
          F.avg("Visibility_mi").alias("avg_visibility_mi"),
      )
      .orderBy("year", "State", "Severity")
)

(
    summary.coalesce(1)
           .write.mode("overwrite")
           .option("header", True)
           .csv(OUT_ANALYTICS)
)

print(f"✅ Summary CSV written to: {OUT_ANALYTICS}")

# ------------ (B) DATASET UNTUK ML ------------
ml_cols = [
    "year",
    "month",          # di script 2 kamu month = "yyyy-MM"
    "hour",
    "State",
    "Severity",
    "Distance_mi",
    "Temperature_F",
    "Humidity_pct",
    "Pressure_in",
    "Visibility_mi",
    "Wind_Speed_mph",
    "Precipitation_in"
]

existing = set(df.columns)
ml_cols = [c for c in ml_cols if c in existing]

ml_df = df.select(*ml_cols)

must_have = [c for c in ["Severity", "Distance_mi"] if c in ml_cols]
ml_df = ml_df.dropna(subset=must_have)

(
    ml_df.write.mode("overwrite")
         .parquet(OUT_ML)
)

print(f"ML Parquet written to: {OUT_ML}")

spark.stop()
