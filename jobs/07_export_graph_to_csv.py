# jobs/07_graph_export.py
import os
import json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ========= PATHS =========
INPUT_CURATED = os.getenv("INPUT_CURATED", "/data/data_lake/curated/us_accidents_features")
OUT_GRAPH = os.getenv("OUT_GRAPH", "/data/data_lake/serving/graph_csv/us_accidents_graph_csv")

TOPN_HUBS = int(os.getenv("TOPN_HUBS", "300"))
TOPN_EDGES = int(os.getenv("TOPN_EDGES", "300"))
TOPN_COMM = int(os.getenv("TOPN_COMM", "300"))

# Edge threshold (biar gak kosong)
MIN_COOC = int(os.getenv("MIN_COOC", "2"))

# Pilihan bucket untuk city↔city co-occurrence (top_edges)
# hour / weather / month / all
BUCKET_MODE = os.getenv("BUCKET_MODE", "weather").lower()


def write_csv(df, out_dir, write_mode="overwrite"):
    (
        df.coalesce(1)
          .write.mode(write_mode)
          .option("header", True)
          .csv(out_dir)
    )


def ensure_cols(df, cols_with_type):
    """
    cols_with_type: list of (colname, sparkTypeString)
    kalau gak ada kolomnya -> buat null
    """
    for c, t in cols_with_type:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(t))
    return df


def make_bucket(df):
    if BUCKET_MODE == "hour":
        if "hour" in df.columns:
            return df.withColumn("bucket", F.col("hour").cast("int"))
        if "Start_Time" in df.columns:
            return df.withColumn("bucket", F.hour(F.to_timestamp("Start_Time")))
        return df.withColumn("bucket", F.lit("all"))

    if BUCKET_MODE == "month":
        if "month" in df.columns:
            return df.withColumn("bucket", F.col("month").cast("string"))
        if "month_num" in df.columns:
            return df.withColumn("bucket", F.col("month_num").cast("int"))
        return df.withColumn("bucket", F.lit("all"))

    if BUCKET_MODE == "weather":
        if "weather_bucket" in df.columns:
            return df.withColumn("bucket", F.col("weather_bucket").cast("string"))
        if "Weather_Condition" in df.columns:
            return df.withColumn("bucket", F.col("Weather_Condition").cast("string"))
        return df.withColumn("bucket", F.lit("all"))

    return df.withColumn("bucket", F.lit("all"))


def build_bipartite_edges(df, left_col, right_col, out_folder_name, min_cooc, topn):
    """
    Buat edges bipartite: left_col ↔ right_col
    Output schema minimal yang dibaca Streamlit:
      src, dst, cooc
    """
    if left_col not in df.columns or right_col not in df.columns:
        print(f"⚠️ Skip {out_folder_name}: missing {left_col} or {right_col}")
        return

    tmp = df.select(
        F.trim(F.col(left_col).cast("string")).alias("src"),
        F.trim(F.col(right_col).cast("string")).alias("dst"),
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
           .limit(topn)
    )

    out_dir = os.path.join(OUT_GRAPH, out_folder_name)
    write_csv(edges, out_dir)
    print("✅ Graph CSV:", out_dir)


def main():
    spark = (
        SparkSession.builder
        .appName("07_graph_export")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print("INPUT_CURATED:", INPUT_CURATED)
    print("OUT_GRAPH    :", OUT_GRAPH)
    print("BUCKET_MODE  :", BUCKET_MODE)
    print("TOPN_EDGES   :", TOPN_EDGES)
    print("TOPN_HUBS    :", TOPN_HUBS)
    print("TOPN_COMM    :", TOPN_COMM)
    print("MIN_COOC     :", MIN_COOC)

    df = spark.read.parquet(INPUT_CURATED)

    # ====== pastikan kolom ada ======
    df = ensure_cols(df, [
        ("State", "string"),
        ("City", "string"),
        ("weather_bucket", "string"),
        ("Weather_Condition", "string"),
        ("year", "int"),
        ("month", "string"),
        ("month_num", "int"),
        ("hour", "int"),
        ("dow", "int"),
        ("Start_Time", "string"),
    ])

    # ====== normalize + fill ======
    g = (
        df.select(
            F.trim(F.col("State")).alias("State"),
            F.trim(F.col("City")).alias("City"),
            F.trim(F.col("weather_bucket")).alias("weather_bucket"),
            F.trim(F.col("Weather_Condition")).alias("Weather_Condition"),
            F.col("year").cast("int").alias("year"),
            F.col("month").cast("string").alias("month"),
            F.col("month_num").cast("int").alias("month_num"),
            F.col("hour").cast("int").alias("hour"),
            F.col("dow").cast("int").alias("dow"),
            F.col("Start_Time").cast("string").alias("Start_Time"),
        )
        .fillna({"State": "UNKNOWN", "City": "UNKNOWN"})
    )

    # buat bucket (weather/hour/month/all) untuk top_edges (city↔city)
    g = make_bucket(g).fillna({"bucket": "all"})

    # buang UNKNOWN biar meaningful, tapi jangan sampai kosong
    g2 = g.filter((F.col("State") != "UNKNOWN") & (F.col("City") != "UNKNOWN"))
    if g2.limit(1).count() == 0:
        g2 = g

    # =========================================================
    # A) City ↔ City co-occurrence (top_edges) + hubs + communities
    # =========================================================
    g2 = g2.withColumn("node", F.concat_ws("|", F.col("State"), F.col("City")))

    grp = g2.groupBy("State", "bucket").agg(F.collect_set("node").alias("nodes"))

    a = grp.select("State", "bucket", F.explode("nodes").alias("src"), F.col("nodes").alias("nodes"))
    b = a.select("State", "bucket", "src", F.explode("nodes").alias("dst"))
    pairs = b.filter(F.col("src") < F.col("dst"))

    edges_full = (
        pairs.groupBy("src", "dst")
             .agg(F.count("*").alias("cooc"))
             .filter(F.col("cooc") >= F.lit(MIN_COOC))
             .orderBy(F.desc("cooc"))
    )

    top_edges = edges_full.limit(TOPN_EDGES)

    deg_src = edges_full.groupBy("src").agg(
        F.count("*").alias("degree"),
        F.sum("cooc").alias("degree_weighted")
    )
    deg_dst = edges_full.groupBy("dst").agg(
        F.count("*").alias("degree"),
        F.sum("cooc").alias("degree_weighted")
    )

    nodes = (
        deg_src.select(F.col("src").alias("node"), "degree", "degree_weighted")
        .unionByName(deg_dst.select(F.col("dst").alias("node"), "degree", "degree_weighted"))
        .groupBy("node")
        .agg(F.sum("degree").alias("degree"), F.sum("degree_weighted").alias("degree_weighted"))
        .orderBy(F.desc("degree_weighted"))
        .limit(TOPN_HUBS)
    )

    nodes = (
        nodes.withColumn("State", F.split(F.col("node"), r"\|").getItem(0))
             .withColumn("City", F.split(F.col("node"), r"\|").getItem(1))
             .select("State", "City", "degree", "degree_weighted")
    )

    top_edges = (
        top_edges
        .withColumn("src_state", F.split(F.col("src"), r"\|").getItem(0))
        .withColumn("src_city",  F.split(F.col("src"), r"\|").getItem(1))
        .withColumn("dst_state", F.split(F.col("dst"), r"\|").getItem(0))
        .withColumn("dst_city",  F.split(F.col("dst"), r"\|").getItem(1))
        .select("src", "dst", "src_state", "src_city", "dst_state", "dst_city", "cooc")
        .orderBy(F.desc("cooc"))
    )

    comm_sizes = (
        g2.groupBy("State")
          .agg(F.countDistinct("City").alias("n_nodes"))
          .orderBy(F.desc("n_nodes"))
          .limit(TOPN_COMM)
          .withColumnRenamed("State", "label")
    )

    os.makedirs(OUT_GRAPH, exist_ok=True)

    write_csv(nodes, os.path.join(OUT_GRAPH, "top_hubs_nodes"))
    write_csv(top_edges, os.path.join(OUT_GRAPH, "top_edges"))
    write_csv(comm_sizes, os.path.join(OUT_GRAPH, "community_sizes"))

    print("✅ Graph CSV written:", OUT_GRAPH)
    print("   - top_hubs_nodes")
    print("   - top_edges")
    print("   - community_sizes")

    # =========================================================
    # B) Bipartite edge folders expected by Streamlit (script 08)
    # =========================================================
    # City ↔ Hour
    build_bipartite_edges(g2, "City", "hour", "edges_city_hour", MIN_COOC, TOPN_EDGES)

    # City ↔ Weather (prefer weather_bucket, fallback Weather_Condition)
    # Use whichever has more non-null values
    wb_nonnull = g2.filter(F.col("weather_bucket").isNotNull() & (F.col("weather_bucket") != "")).limit(1).count()
    if wb_nonnull > 0:
        build_bipartite_edges(g2, "City", "weather_bucket", "edges_city_weather", MIN_COOC, TOPN_EDGES)
        # Hour ↔ Weather
        build_bipartite_edges(g2, "hour", "weather_bucket", "edges_hour_weather", MIN_COOC, TOPN_EDGES)
    else:
        build_bipartite_edges(g2, "City", "Weather_Condition", "edges_city_weather", MIN_COOC, TOPN_EDGES)
        build_bipartite_edges(g2, "hour", "Weather_Condition", "edges_hour_weather", MIN_COOC, TOPN_EDGES)

    # City ↔ DayOfWeek
    build_bipartite_edges(g2, "City", "dow", "edges_city_dow", MIN_COOC, TOPN_EDGES)

    # meta (optional)
    meta = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "input_curated": INPUT_CURATED,
        "out_graph": OUT_GRAPH,
        "bucket_mode": BUCKET_MODE,
        "min_cooc": MIN_COOC,
        "topn_edges": TOPN_EDGES,
        "topn_hubs": TOPN_HUBS,
        "topn_comm": TOPN_COMM,
        "streamlit_expected_folders": [
            "top_edges",
            "top_hubs_nodes",
            "community_sizes",
            "edges_city_weather",
            "edges_city_hour",
            "edges_hour_weather",
            "edges_city_dow",
        ],
    }
    meta_out = os.path.join(OUT_GRAPH, "_run_meta.json")
    spark.createDataFrame([(json.dumps(meta),)], ["json"]).coalesce(1).write.mode("overwrite").text(meta_out)
    print("✅ Meta written:", meta_out)

    spark.stop()


if __name__ == "__main__":
    main()
