from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    hour,
    month,
    dayofweek,
    to_timestamp,
    log1p
)

spark = SparkSession.builder \
    .appName("ASHRAE_Cleaning_Feature_Engineering") \
    .getOrCreate()

print("Loading datasets...")

# Load datasets
train_df = spark.read.csv("data/raw/train.csv", header=True, inferSchema=True)
weather_df = spark.read.csv("data/raw/weather_train.csv", header=True, inferSchema=True)
building_df = spark.read.csv("data/raw/building_metadata.csv", header=True, inferSchema=True)

print("Converting timestamp column...")

# Convert timestamp to proper timestamp type
train_df = train_df.withColumn("timestamp", to_timestamp(col("timestamp")))
weather_df = weather_df.withColumn("timestamp", to_timestamp(col("timestamp")))

print("Joining datasets...")

# Join datasets
df = train_df.join(building_df, "building_id", "left") \
             .join(weather_df, ["site_id", "timestamp"], "left")

print("Basic cleaning...")

# Drop rows where target is null
df = df.dropna(subset=["meter_reading"])

# Fill weather nulls (very common in ASHRAE)
df = df.fillna({
    "air_temperature": 0,
    "cloud_coverage": 0
})

print("Feature engineering...")

# Extract time features
df = df.withColumn("hour", hour(col("timestamp"))) \
       .withColumn("month", month(col("timestamp"))) \
       .withColumn("dayofweek", dayofweek(col("timestamp")))

# Log transform target (VERY IMPORTANT for this dataset)
df = df.withColumn("meter_reading", log1p(col("meter_reading")))

print("Selecting final columns...")

df = df.select(
    "building_id",
    "meter",
    "square_feet",
    "air_temperature",
    "cloud_coverage",
    "hour",
    "month",
    "dayofweek",
    "meter_reading"
)

print("Writing parquet file...")

# Save optimized parquet
df.write.mode("overwrite").parquet("data/processed/clean_data.parquet")

spark.stop()

print("Spark preprocessing with feature engineering complete 🚀")