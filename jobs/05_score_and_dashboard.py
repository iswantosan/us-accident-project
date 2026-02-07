# jobs/05_score_and_dashboard_ready.py
import os
import json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel

# ========= PATHS =========
INPUT_CURATED = os.getenv("INPUT_CURATED", "/data/data_lake/curated/us_accidents_features")
MODEL_IN = os.getenv("MODEL_IN", "/data/models/us_accidents_severity_rf")

OUT_SCORED = os.getenv("OUT_SCORED", "/data/data_lake/serving/scored/us_accidents_scored_parquet")
OUT_DASHBOARD = os.getenv("OUT_DASHBOARD", "/data/data_lake/serving/dashboard/us_accidents_dashboard_csv")

WRITE_MODE = os.getenv("WRITE_MODE", "overwrite")
SCORE_SAMPLE_N = int(os.getenv("SCORE_SAMPLE_N", "0"))  # 0=full

# Turn ON to print confusion + mismatch stats (can be expensive on full data)
DEBUG_MISMATCH = os.getenv("DEBUG_MISMATCH", "1") == "1"
# Limit rows for debug confusion table (avoid huge show)
DEBUG_CONFUSION_LIMIT = int(os.getenv("DEBUG_CONFUSION_LIMIT", "200"))

# Safety: avoid calling count() twice on huge DF in debug
DEBUG_SAMPLE_FOR_MISMATCH = int(os.getenv("DEBUG_SAMPLE_FOR_MISMATCH", "0"))  # 0=use full scored_out


def ensure_month_num(df):
    """
    Training pipeline butuh month_num.
    Tries multiple sources and ensures non-null int to reduce pipeline failures.
    """
    if "month_num" in df.columns:
        return df

    # option 1: month string like "yyyy-MM"
    if "month" in df.columns:
        return df.withColumn("month_num", F.substring(F.col("month"), 6, 2).cast("int"))

    # option 2: curated timestamp
    if "Start_Time_ts" in df.columns:
        return df.withColumn("month_num", F.month(F.col("Start_Time_ts")))

    # option 3: parse Start_Time string
    if "Start_Time" in df.columns:
        tmp = df.withColumn("Start_Time_ts_tmp", F.to_timestamp("Start_Time"))
        return tmp.withColumn("month_num", F.month(F.col("Start_Time_ts_tmp"))).drop("Start_Time_ts_tmp")

    # fallback: set to 1 (NOT null) so pipeline/indexers don't choke
    return df.withColumn("month_num", F.lit(1).cast("int"))


def get_label_indexer_model(pipeline_model: PipelineModel):
    """
    Find StringIndexerModel that produces outputCol == 'label'.
    This is critical to map prediction index -> original label values.
    """
    for st in pipeline_model.stages:
        if st.__class__.__name__ == "StringIndexerModel":
            try:
                if st.getOutputCol() == "label":
                    return st
            except Exception:
                pass
    return None


def map_pred_label_to_severity(scored_df, pipeline_model: PipelineModel):
    """
    Robust mapping:
      pred_label (0..K-1) -> label_indexer.labels[i] (e.g., "1","2","3","4") -> int severity
    Fallback: pred_label + 1
    """
    label_stage = get_label_indexer_model(pipeline_model)
    if label_stage is None:
        return scored_df.withColumn("pred_severity", (F.col("pred_label") + F.lit(1)).cast("int"))

    labels = list(label_stage.labels)  # list[str], order corresponds to indices
    expr = None
    for i, s in enumerate(labels):
        try:
            sev_int = int(s)
        except Exception:
            sev_int = None

        cond = (F.col("pred_label") == F.lit(i))
        if expr is None:
            expr = F.when(cond, F.lit(sev_int).cast("int"))
        else:
            expr = expr.when(cond, F.lit(sev_int).cast("int"))

    expr = expr.otherwise(F.lit(None).cast("int"))
    return scored_df.withColumn("pred_severity", expr)


def write_meta_text(spark, out_path, meta: dict, write_mode: str):
    """
    Writes meta as Spark text output (a folder containing part-*.txt).
    """
    spark.createDataFrame([(json.dumps(meta),)], ["json"]).coalesce(1).write.mode(write_mode).text(out_path)


def main():
    spark = (
        SparkSession.builder
        .appName("05_score_and_dashboard_ready")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("INPUT_CURATED :", INPUT_CURATED)
    print("MODEL_IN      :", MODEL_IN)
    print("OUT_SCORED    :", OUT_SCORED)
    print("OUT_DASHBOARD :", OUT_DASHBOARD)
    print("WRITE_MODE    :", WRITE_MODE)
    print("SCORE_SAMPLE_N:", SCORE_SAMPLE_N)
    print("DEBUG_MISMATCH:", DEBUG_MISMATCH)

    df = spark.read.parquet(INPUT_CURATED)

    # optional sample
    if SCORE_SAMPLE_N > 0:
        df = df.orderBy(F.rand(7)).limit(SCORE_SAMPLE_N)

    # ensure required feature
    df = ensure_month_num(df)

    model = PipelineModel.load(MODEL_IN)
    scored = model.transform(df)

    # prediction index
    scored = scored.withColumn("pred_label", F.col("prediction").cast("int"))

    # confidence
    try:
        from pyspark.ml.functions import vector_to_array
        scored = scored.withColumn("prob_arr", vector_to_array("probability"))
        scored = scored.withColumn("pred_conf", F.array_max("prob_arr"))
    except Exception:
        scored = scored.withColumn("pred_conf", F.lit(None).cast("double"))

    # robust mapping prediction -> real severity values
    scored = map_pred_label_to_severity(scored, model)

    # columns to keep
    keep = [
        "ID",
        "Severity",          # actual / ground truth
        "pred_severity",     # predicted
        "pred_conf",
        "State", "County", "City", "Zipcode",
        "year", "month", "month_num", "dow", "hour",
        "weather_bucket", "Weather_Condition",
        "Distance_mi",
        "Temperature_F", "Humidity_pct", "Pressure_in", "Visibility_mi",
        "Wind_Speed_mph", "Precipitation_in",
        "duration_min",
        "is_weekend", "is_night",
    ]
    keep = [c for c in keep if c in scored.columns]
    scored_out = scored.select(*keep)

    # write scored parquet
    scored_out.write.mode(WRITE_MODE).parquet(OUT_SCORED)
    print("✅ Scored parquet written:", OUT_SCORED)

    # ===== DEBUG: mismatch check =====
    if DEBUG_MISMATCH and all(c in scored_out.columns for c in ["Severity", "pred_severity"]):
        dbg = scored_out
        if DEBUG_SAMPLE_FOR_MISMATCH > 0:
            dbg = dbg.orderBy(F.rand(13)).limit(DEBUG_SAMPLE_FOR_MISMATCH)
            print(f"🔎 DEBUG_SAMPLE_FOR_MISMATCH enabled: {DEBUG_SAMPLE_FOR_MISMATCH}")

        total = dbg.count()
        mismatch = dbg.filter(F.col("Severity") != F.col("pred_severity")).count()
        pct = (mismatch / total * 100.0) if total else 0.0
        print(f"🔎 Mismatch rows: {mismatch} / {total} ({pct:.2f}%)")

        print("🔎 Confusion counts (sample or full depending on DEBUG_SAMPLE_FOR_MISMATCH):")
        (
            dbg.groupBy("Severity", "pred_severity")
            .count()
            .orderBy("Severity", "pred_severity")
            .show(DEBUG_CONFUSION_LIMIT, truncate=False)
        )

    # ========== DASHBOARD AGG (CSV) ==========
    def write_csv(df2, name: str):
        out = f"{OUT_DASHBOARD}/{name}"
        (
            df2.coalesce(1)
            .write.mode(WRITE_MODE)
            .option("header", True)
            .csv(out)
        )
        print("✅ Dashboard CSV:", out)

    # ------------------------
    # PREDICTED AGGREGATIONS
    # ------------------------
    if all(c in scored_out.columns for c in ["year", "month", "pred_severity"]):
        trend_pred = (
            scored_out.groupBy("year", "month", "pred_severity")
            .agg(
                F.count("*").alias("n"),
                F.avg("pred_conf").alias("avg_conf"),
            )
            .orderBy("year", "month", "pred_severity")
        )
        write_csv(trend_pred, "trend_month_pred_severity")

    if all(c in scored_out.columns for c in ["State", "City", "pred_severity"]):
        hotspots_pred = (
            scored_out.groupBy("State", "City")
            .agg(
                F.count("*").alias("n"),
                F.avg("pred_severity").alias("avg_pred_sev"),
                F.avg("pred_conf").alias("avg_conf"),
            )
            .withColumn("risk_score", F.col("n") * F.col("avg_pred_sev"))
            .orderBy(F.desc("risk_score"))
            .limit(3000)
        )
        write_csv(hotspots_pred, "hotspots_city_risk")

    if all(c in scored_out.columns for c in ["weather_bucket", "pred_severity"]):
        by_weather_pred = (
            scored_out.groupBy("weather_bucket", "pred_severity")
            .agg(F.count("*").alias("n"))
            .orderBy("weather_bucket", "pred_severity")
        )
        write_csv(by_weather_pred, "severity_by_weather")

    if all(c in scored_out.columns for c in ["hour", "pred_severity"]):
        by_hour_pred = (
            scored_out.groupBy("hour", "pred_severity")
            .agg(F.count("*").alias("n"))
            .orderBy("hour", "pred_severity")
        )
        write_csv(by_hour_pred, "severity_by_hour")

    # ------------------------
    # ACTUAL (GROUND TRUTH) AGGREGATIONS
    # ------------------------
    if all(c in scored_out.columns for c in ["year", "month", "Severity"]):
        trend_actual = (
            scored_out.groupBy("year", "month", "Severity")
            .agg(F.count("*").alias("n"))
            .orderBy("year", "month", "Severity")
        )
        write_csv(trend_actual, "trend_month_actual_severity")

    if all(c in scored_out.columns for c in ["State", "City", "Severity"]):
        hotspots_actual = (
            scored_out.groupBy("State", "City")
            .agg(
                F.count("*").alias("n"),
                F.avg("Severity").alias("avg_sev"),
            )
            .withColumn("risk_score", F.col("n") * F.col("avg_sev"))
            .orderBy(F.desc("risk_score"))
            .limit(3000)
        )
        write_csv(hotspots_actual, "hotspots_city_actual")

    if all(c in scored_out.columns for c in ["weather_bucket", "Severity"]):
        by_weather_actual = (
            scored_out.groupBy("weather_bucket", "Severity")
            .agg(F.count("*").alias("n"))
            .orderBy("weather_bucket", "Severity")
        )
        write_csv(by_weather_actual, "severity_by_weather_actual")

    if all(c in scored_out.columns for c in ["hour", "Severity"]):
        by_hour_actual = (
            scored_out.groupBy("hour", "Severity")
            .agg(F.count("*").alias("n"))
            .orderBy("hour", "Severity")
        )
        write_csv(by_hour_actual, "severity_by_hour_actual")

    # meta
    meta = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "input_curated": INPUT_CURATED,
        "model_in": MODEL_IN,
        "out_scored": OUT_SCORED,
        "out_dashboard": OUT_DASHBOARD,
        "score_sample_n": SCORE_SAMPLE_N,
        "write_mode": WRITE_MODE,
        "debug_mismatch": DEBUG_MISMATCH,
        "debug_confusion_limit": DEBUG_CONFUSION_LIMIT,
        "debug_sample_for_mismatch": DEBUG_SAMPLE_FOR_MISMATCH,
    }
    meta_out = f"{OUT_DASHBOARD}/_run_meta.json"
    write_meta_text(spark, meta_out, meta, WRITE_MODE)
    print("✅ Meta written:", meta_out)

    spark.stop()


if __name__ == "__main__":
    main()
