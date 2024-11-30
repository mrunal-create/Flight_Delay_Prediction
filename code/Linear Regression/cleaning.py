import os
import time
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml.feature import StringIndexer

# Define a function to clean data
def clean(df):
    """
    Cleans the input DataFrame by:
    - Imputing missing values in numerical columns with the mean if less than 40% missing.
    - Dropping numerical columns with more than 40% missing values.
    - Encoding categorical columns using StringIndexer.
    """
    drop_threshold = 0.6  # 60% missing threshold for dropping columns
    row_drop_threshold = 0.05  # 5% missing threshold for dropping rows
    total_rows = df.count()

    # Identify columns
    numerical_cols = [col_name for col_name, dtype in df.dtypes if dtype in ('int', 'double', 'float')]
    categorical_cols = [col_name for col_name, dtype in df.dtypes if dtype == 'string']

    drop_col_threshold = drop_threshold * total_rows
    row_drop_col_threshold = row_drop_threshold * total_rows

    # Handle numerical columns
    for col_name in numerical_cols:
        missing_count = df.filter(col(col_name).isNull()).count()
        if missing_count > drop_col_threshold:
            df = df.drop(col_name)  # Drop column
        elif missing_count <= row_drop_col_threshold:
            df = df.filter(col(col_name).isNotNull())  # Drop rows
        else:
            mean_value = df.select(mean(col(col_name))).collect()[0][0]
            df = df.withColumn(col_name, when(col(col_name).isNull(), mean_value).otherwise(col(col_name)))

    # Handle categorical columns
    for col_name in categorical_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed", handleInvalid="skip")
        df = indexer.fit(df).transform(df)
        df = df.drop(col_name).withColumnRenamed(f"{col_name}_indexed", col_name)
    
    return df

# File path
file_path = "concatenated_data.csv"
parquet_path = "concatenated_data.parquet"

# Initialize SparkSession with optimized parallelism configuration
spark = SparkSession.builder \
    .appName('Optimized Data Cleaning') \
    .config('spark.executor.memory', '8g') \
    .config('spark.executor.cores', '4') \
    .config('spark.driver.memory', '4g') \
    .config('spark.sql.shuffle.partitions', '200') \
    .getOrCreate()

# Load data
if not os.path.exists(parquet_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    df.write.parquet(parquet_path)
df = spark.read.parquet(parquet_path)

# Start timing
start_time = time.time()

# Clean the DataFrame
clean_df = clean(df)

# Save the cleaned DataFrame to a Parquet file
clean_df.write.parquet("cleaned_data.parquet", mode="overwrite")

# End timing
end_time = time.time()

# Calculate and print cleaning time
cleaning_time = end_time - start_time
print(f"Data cleaning completed in {cleaning_time:.2f} seconds.")

# Stop the Spark session
spark.stop()
