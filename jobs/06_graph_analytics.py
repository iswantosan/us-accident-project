# jobs/06_graph_analytics.py
import os
import json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ========= PATHS =========
SCORED_IN = os.getenv("SCORED_IN", "/data/data_lake/serving/scored/us_accidents_scored_parquet")

GRAPH_CSV_OUT = os.getenv(
    "GRAPH_CSV_OUT",
    "/data/data_lake/serving/graph_csv/us_accidents_graph_csv"
)

WRITE_MODE = os.getenv("WRITE_MODE", "overwrite")

# ========= PARAMS =========
MIN_COOC = int(os.getenv("MIN_COOC", "30"))
TOP_EDGES = int(os.getenv("TOP_EDGES", "3000"))
TOP_HUBS = int(os.getenv("TOP_HUBS", "3000"))
MAX_ITERS = int(os.getenv("MAX_ITERS", "8"))

# For default graph "top_edges" (city↔city)
BUCKET_MODE = os.getenv("BUCKET_MODE", "hour").lower()

USE_PRED_SEV = os.getenv("USE_PRED_SEV", "1") == "1"


def write_csv(df, name: str):
    """
    Write Spark DF to Spark-style CSV folder:
      <GRAPH_CSV_OUT>/<name>/part-*.csv + _SUCCESS
    """
    out = f"{GRAPH_CSV_OUT}/{name}"
    (
        df.coalesce(1)
          .write.mode(WRITE_MODE)
          .option("header", True)
          .csv(out)
    )
    print("✅ Graph CSV:", out)


def ensure_cols(df, cols_with_type):
    for c, t in cols_with_type:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(t))
    return df


def ensure_bucket(df):
    """bucket for city↔city co-occurrence graph (top_edges)."""
    if BUCKET_MODE == "weather":
        if "weather_bucket" in df.columns:
            return df.withColumn("bucket", F.col("weather_bucket").cast("string"))
        elif "Weather_Condition" in df.columns:
            return df.withColumn("bucket", F.col("Weather_Condition").cast("string"))
        return df.withColumn("bucket", F.lit("all"))

    if BUCKET_MODE == "month":
        if "month" in df.columns:
            return df.withColumn("bucket", F.col("month").cast("string"))
        if "month_num" in df.columns:
            return df.withColumn("bucket", F.col("month_num").cast("int"))
        return df.withColumn("bucket", F.lit("all"))

    if BUCKET_MODE == "hour":
        if "hour" in df.columns:
            return df.withColumn("bucket", F.col("hour").cast("int"))
        return df.withColumn("bucket", F.lit("all"))

    return df.withColumn("bucket", F.lit("all"))


def split_node(df_in, node_col="node"):
    return (
        df_in.withColumn("state", F.split(F.col(node_col), r"\|").getItem(0))
             .withColumn("city", F.split(F.col(node_col), r"\|").getItem(1))
    )


def build_bipartite_edges(df, left_col, right_col, out_name, min_cooc=30, top_edges=3000):
    """
    Generic bipartite edges:
      src = left_col
      dst = right_col
      weight = cooc = count(*)
    Output columns: src, dst, cooc (+ optional state/year if later you need)
    """
    if left_col not in df.columns or right_col not in df.columns:
        print(f"⚠️ Skip {out_name}: missing columns {left_col} or {right_col}")
        return

    tmp = df.select(
        F.trim(F.col(left_col)).alias("src"),
        F.trim(F.col(right_col)).alias("dst")
    )

    tmp = tmp.filter(
        F.col("src").isNotNull() & (F.col("src") != "") &
        F.col("dst").isNotNull() & (F.col("dst") != "")
    )

    edges = (
        tmp.groupBy("src", "dst")
           .agg(F.count("*").alias("cooc"))
           .filter(F.col("cooc") >= F.lit(min_cooc))
           .orderBy(F.desc("cooc"))
           .limit(top_edges)
    )

    write_csv(edges, out_name)


def main():
    spark = (
        SparkSession.builder
        .appName("06_graph_analytics")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("SCORED_IN     :", SCORED_IN)
    print("GRAPH_CSV_OUT :", GRAPH_CSV_OUT)
    print("WRITE_MODE    :", WRITE_MODE)
    print("BUCKET_MODE   :", BUCKET_MODE)
    print("MIN_COOC      :", MIN_COOC)
    print("TOP_EDGES     :", TOP_EDGES)
    print("TOP_HUBS      :", TOP_HUBS)
    print("MAX_ITERS     :", MAX_ITERS)
    print("USE_PRED_SEV  :", USE_PRED_SEV)

    df = spark.read.parquet(SCORED_IN)

    # make sure needed cols exist
    df = ensure_cols(df, [
        ("State", "string"),
        ("City", "string"),
        ("hour", "int"),
        ("dow", "int"),
        ("weather_bucket", "string"),
        ("Weather_Condition", "string"),
        ("month", "string"),
        ("month_num", "int"),
        ("pred_severity", "int"),
        ("Severity", "int"),
        ("year", "int"),
    ])

    # =========================================================
    # 1) DEFAULT graph: City ↔ City co-occurrence => top_edges
    # =========================================================

    # required columns
    for c in ["State", "City"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing required col: {c}. Available: {df.columns}")

    # bucket for city-city graph
    base_df = ensure_bucket(df)

    # node = State|City
    base_df = base_df.withColumn("node", F.concat_ws("|", F.col("State"), F.col("City")))

    keep = ["State", "bucket", "node"]
    if USE_PRED_SEV and ("pred_severity" in base_df.columns):
        keep.append("pred_severity")

    base = base_df.select(*keep)

    grp = base.groupBy("State", "bucket").agg(F.collect_set("node").alias("nodes"))

    a = grp.select("State", "bucket", F.explode("nodes").alias("src"), F.col("nodes").alias("nodes"))
    b = a.select("State", "bucket", "src", F.explode("nodes").alias("dst"))
    pairs = b.filter(F.col("src") < F.col("dst"))

    edges_city_city = (
        pairs.groupBy("src", "dst")
             .agg(F.count("*").alias("cooc"))
             .filter(F.col("cooc") >= F.lit(MIN_COOC))
             .orderBy(F.desc("cooc"))
             .limit(TOP_EDGES)
    )

    # node risk
    if USE_PRED_SEV and ("pred_severity" in base.columns):
        node_risk = base.groupBy("node").agg(
            F.count("*").alias("n"),
            F.avg("pred_severity").alias("avg_pred_sev")
        )

        edges_city_city = (
            edges_city_city.join(
                node_risk.select(F.col("node").alias("src"), F.col("avg_pred_sev").alias("src_avg_sev")),
                on="src",
                how="left"
            ).join(
                node_risk.select(F.col("node").alias("dst"), F.col("avg_pred_sev").alias("dst_avg_sev")),
                on="dst",
                how="left"
            ).withColumn(
                "edge_risk",
                (F.col("src_avg_sev") + F.col("dst_avg_sev")) / F.lit(2.0)
            )
        )
    else:
        node_risk = base.groupBy("node").agg(F.count("*").alias("n"))

    # degrees
    deg_src = edges_city_city.groupBy("src").agg(F.sum("cooc").alias("deg_weighted"), F.count("*").alias("deg"))
    deg_dst = edges_city_city.groupBy("dst").agg(F.sum("cooc").alias("deg_weighted"), F.count("*").alias("deg"))

    degrees = (
        deg_src.select(F.col("src").alias("node"), "deg", "deg_weighted")
        .unionByName(deg_dst.select(F.col("dst").alias("node"), "deg", "deg_weighted"))
        .groupBy("node")
        .agg(F.sum("deg").alias("degree"), F.sum("deg_weighted").alias("degree_weighted"))
        .join(node_risk, "node", "left")
        .orderBy(F.desc("degree_weighted"))
        .limit(TOP_HUBS)
    )

    # communities (cheap label propagation)
    labels = degrees.select("node").withColumn("label", F.col("node"))

    undirected = (
        edges_city_city.select(F.col("src").alias("u"), F.col("dst").alias("v"), "cooc")
        .unionByName(edges_city_city.select(F.col("dst").alias("u"), F.col("src").alias("v"), "cooc"))
    )

    for _ in range(MAX_ITERS):
        joined = (
            undirected.join(
                labels.select(F.col("node").alias("v"), F.col("label").alias("v_label")),
                on="v",
                how="left"
            ).groupBy("u").agg(F.min("v_label").alias("min_neighbor_label"))
        )

        labels = (
            labels.join(joined, labels.node == joined.u, "left")
                  .withColumn("new_label", F.least(F.col("label"), F.col("min_neighbor_label")))
                  .select(labels.node.alias("node"), F.col("new_label").alias("label"))
        )

    # beautify outputs
    edges_out = (
        edges_city_city.withColumn("src_state", F.split(F.col("src"), r"\|").getItem(0))
                       .withColumn("src_city",  F.split(F.col("src"), r"\|").getItem(1))
                       .withColumn("dst_state", F.split(F.col("dst"), r"\|").getItem(0))
                       .withColumn("dst_city",  F.split(F.col("dst"), r"\|").getItem(1))
                       .orderBy(F.desc("cooc"))
    )

    hubs_out = split_node(degrees, "node").select(
        "state", "city", "degree", "degree_weighted",
        *([c for c in ["n", "avg_pred_sev"] if c in degrees.columns])
    ).orderBy(F.desc("degree_weighted"))

    comm_out = split_node(labels, "node").select("state", "city", "label")

    # write default outputs used by Streamlit
    write_csv(edges_out, "top_edges")
    write_csv(hubs_out, "top_hubs_nodes")
    write_csv(comm_out, "community_sizes")

    # =========================================================
    # 2) Bipartite edges expected by Streamlit script 08
    # =========================================================
    # City ↔ Hour
    build_bipartite_edges(df, "City", "hour", "edges_city_hour", min_cooc=MIN_COOC, top_edges=TOP_EDGES)

    # City ↔ Weather
    # prefer weather_bucket, fallback Weather_Condition
    if df.filter(F.col("weather_bucket").isNotNull()).limit(1).count() > 0:
        build_bipartite_edges(df, "City", "weather_bucket", "edges_city_weather", min_cooc=MIN_COOC, top_edges=TOP_EDGES)
    else:
        build_bipartite_edges(df, "City", "Weather_Condition", "edges_city_weather", min_cooc=MIN_COOC, top_edges=TOP_EDGES)

    # Hour ↔ Weather
    if df.filter(F.col("weather_bucket").isNotNull()).limit(1).count() > 0:
        build_bipartite_edges(df, "hour", "weather_bucket", "edges_hour_weather", min_cooc=MIN_COOC, top_edges=TOP_EDGES)
    else:
        build_bipartite_edges(df, "hour", "Weather_Condition", "edges_hour_weather", min_cooc=MIN_COOC, top_edges=TOP_EDGES)

    # City ↔ DayOfWeek
    build_bipartite_edges(df, "City", "dow", "edges_city_dow", min_cooc=MIN_COOC, top_edges=TOP_EDGES)

    # meta
    meta = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "scored_in": SCORED_IN,
        "graph_csv_out": GRAPH_CSV_OUT,
        "min_cooc": MIN_COOC,
        "top_edges": TOP_EDGES,
        "top_hubs": TOP_HUBS,
        "max_iters": MAX_ITERS,
        "bucket_mode": BUCKET_MODE,
        "use_pred_sev": USE_PRED_SEV,
        "streamlit_expected_folders": [
            "top_edges",
            "top_hubs_nodes",
            "edges_city_weather",
            "edges_city_hour",
            "edges_hour_weather",
            "edges_city_dow",
        ]
    }
    meta_out = f"{GRAPH_CSV_OUT}/_run_meta.json"
    spark.createDataFrame([(json.dumps(meta),)], ["json"]).coalesce(1).write.mode(WRITE_MODE).text(meta_out)
    print("✅ Meta written:", meta_out)

    spark.stop()


if __name__ == "__main__":
    main()
