import os
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window

# ====== Paths (di dalam container / Linux path) ======
IN_PARQUET  = "/data/data_lake/processed/us_accidents_parquet"
OUT_PARQUET = "/data/data_lake/curated/us_accidents_features"

def main():
    spark = (
        SparkSession.builder
        .appName("US Accidents - Curate Features")
        .master("local[*]")
        # tuning ringan biar stabil
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024))  # 128MB
        .getOrCreate()
    )

    print(f"Reading: {IN_PARQUET}")
    df = spark.read.parquet(IN_PARQUET)

    # =========================
    # 1) BASIC CLEANSING
    # =========================

    # Cast tipe data penting (banyak dataset csv kebaca string)
    df = (
        df.withColumn("Severity", F.col("Severity").cast("int"))
          .withColumn("Distance_mi", F.col("Distance(mi)").cast("double"))
          .withColumn("Temperature_F", F.col("Temperature(F)").cast("double"))
          .withColumn("Humidity_pct", F.col("Humidity(%)").cast("double"))
          .withColumn("Pressure_in", F.col("Pressure(in)").cast("double"))
          .withColumn("Visibility_mi", F.col("Visibility(mi)").cast("double"))
          .withColumn("Wind_Speed_mph", F.col("Wind_Speed(mph)").cast("double"))
          .withColumn("Precipitation_in", F.col("Precipitation(in)").cast("double"))
    )

    # Parse waktu
    df = df.withColumn("Start_Time_ts", F.to_timestamp("Start_Time"))
    df = df.withColumn("End_Time_ts", F.to_timestamp("End_Time"))

    # Filter record jelek (waktu wajib ada)
    df = df.filter(F.col("Start_Time_ts").isNotNull())

    # Durasi (menit) -> kalau End null, durasi null
    df = df.withColumn(
        "duration_min",
        F.when(F.col("End_Time_ts").isNotNull(),
               (F.col("End_Time_ts").cast("long") - F.col("Start_Time_ts").cast("long")) / 60.0
        ).otherwise(F.lit(None).cast("double"))
    )

    # Handle nilai outlier / invalid sederhana
    # (ini cleansing ringan, bukan perfect)
    df = df.withColumn(
        "duration_min",
        F.when((F.col("duration_min") < 0) | (F.col("duration_min") > 60*24*7), None).otherwise(F.col("duration_min"))
    )

    # Isi nilai null untuk kategori penting
    df = df.fillna({
        "State": "UNKNOWN",
        "City": "UNKNOWN",
        "County": "UNKNOWN",
        "Weather_Condition": "UNKNOWN",
        "Timezone": "UNKNOWN",
        "Source": "UNKNOWN",
        "Country": "UNKNOWN",
    })

    # =========================
    # 2) FEATURE ENGINEERING
    # =========================

    df = (
        df
        .withColumn("year",  F.year("Start_Time_ts"))
        .withColumn("month", F.date_format("Start_Time_ts", "yyyy-MM"))
        .withColumn("dow",   F.date_format("Start_Time_ts", "E"))            # Mon/Tue/...
        .withColumn("hour",  F.hour("Start_Time_ts"))
        .withColumn("is_weekend", F.col("dow").isin(["Sat", "Sun"]))
        .withColumn("is_night", (F.col("Sunrise_Sunset") == F.lit("Night")))
    )

    # Convert kolom boolean string "True/False" -> boolean
    bool_cols = [
        "Amenity","Bump","Crossing","Give_Way","Junction","No_Exit","Railway",
        "Roundabout","Station","Stop","Traffic_Calming","Traffic_Signal","Turning_Loop"
    ]
    for c in bool_cols:
        if c in df.columns:
            df = df.withColumn(c, (F.col(c) == F.lit("True")) | (F.col(c) == F.lit(True)))

    # Simple weather bucket (biar gampang dipakai chart)
    df = df.withColumn(
        "weather_bucket",
        F.when(F.lower(F.col("Weather_Condition")).contains("rain"), "rain")
         .when(F.lower(F.col("Weather_Condition")).contains("snow"), "snow")
         .when(F.lower(F.col("Weather_Condition")).contains("fog"), "fog")
         .when(F.lower(F.col("Weather_Condition")).contains("storm"), "storm")
         .when(F.lower(F.col("Weather_Condition")).contains("clear"), "clear")
         .when(F.lower(F.col("Weather_Condition")).contains("cloud"), "cloudy")
         .otherwise("other")
    )

    # One more: severity bucket
    df = df.withColumn(
        "severity_bucket",
        F.when(F.col("Severity").isNull(), "unknown")
         .when(F.col("Severity") <= 2, "low")
         .when(F.col("Severity") == 3, "medium")
         .otherwise("high")
    )

    # =========================
    # 3) SELECT FINAL COLUMNS (lebih ramping)
    # =========================
    keep = [
        "ID", "Source", "Severity", "severity_bucket",
        "Start_Time_ts", "End_Time_ts", "duration_min",
        "year", "month", "dow", "hour", "is_weekend", "is_night",
        "State", "County", "City", "Zipcode", "Country", "Timezone",
        "Distance_mi",
        "Temperature_F", "Humidity_pct", "Pressure_in", "Visibility_mi",
        "Wind_Direction", "Wind_Speed_mph", "Precipitation_in",
        "Weather_Condition", "weather_bucket",
    ] + [c for c in bool_cols if c in df.columns]

    df2 = df.select([c for c in keep if c in df.columns])

    # =========================
    # 4) WRITE CURATED PARQUET
    # =========================
    # Partition by year/month biar gampang query + gak berat
    print(f"Writing curated parquet to: {OUT_PARQUET}")
    (
        df2
        .repartition("year", "month")  # bikin file tidak 1 jumbo
        .write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(OUT_PARQUET)
    )

    spark.stop()
    print("DONE Curated features written.")

if __name__ == "__main__":
    main()
