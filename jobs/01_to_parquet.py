from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CSV_PATH = "/data/dataset/US_Accidents_2016_2023.csv"
OUT_PATH = "/data/data_lake/processed/us_accidents_parquet"

spark = SparkSession.builder.appName("CSV_to_Parquet").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)
    .csv(CSV_PATH)
)

# Minimal cleaning + time parsing
df2 = (
    df
    .withColumn("Start_Time", F.to_timestamp("Start_Time"))
    .withColumn("End_Time", F.to_timestamp("End_Time"))
    .filter(F.col("Start_Time").isNotNull())
    .filter(F.col("Severity").isNotNull())
)

# Partition untuk performa query
df2 = (
    df2
    .withColumn("year", F.year("Start_Time"))
    .withColumn("month", F.month("Start_Time"))
)

(
    df2.write
    .mode("overwrite")
    .partitionBy("year", "month")
    .parquet(OUT_PATH)
)

print("DONE. Parquet written to:", OUT_PATH)
spark.stop()
