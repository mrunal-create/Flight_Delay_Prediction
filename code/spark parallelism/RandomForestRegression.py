# -*- coding: utf-8 -*-
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import DataFrame
import pandas as pd

# Initialize Spark session with configuration
spark = SparkSession.builder \
    .appName('Parallelism RandomForest') \
    .config('spark.executor.memory', '6g') \
    .config('spark.driver.memory', '6g') \
    .config('spark.sql.shuffle.partitions', '200') \
    .getOrCreate()

#File paths
file_path = "/home/jain.uma/RandomForest/concatenated_data.csv"  
parquet_path = "/home/jain.uma/RandomForest/concatenated_data.parquet"  
cleaned_parquet_path = "/home/jain.uma/RandomForest/clean_flights.parquet" 

# Data cleaning function
def clean(df: DataFrame) -> DataFrame:
    """
    Cleans the input DataFrame by:
    - Imputing missing values for numerical columns with the mean.
    - Dropping rows with too many missing values.
    - Indexing categorical columns using StringIndexer.
    """
    drop_threshold = 0.6  # 60% missing threshold for dropping columns
    row_drop_threshold = 0.05  # 5% missing threshold for dropping rows
    total_rows = df.count()

    # Identifying column types
    numerical_cols = [col_name for col_name, dtype in df.dtypes if dtype in ('int', 'double', 'float')]
    categorical_cols = [col_name for col_name, dtype in df.dtypes if dtype == 'string']

    row_drop_col_threshold = row_drop_threshold * total_rows

    # Handle missing values for numerical columns
    for col_name in numerical_cols:
        missing_count = df.filter(col(col_name).isNull()).count()
        if missing_count > drop_threshold * total_rows:
            df = df.drop(col_name)  # Drop column if missing data exceeds threshold

        elif missing_count <= row_drop_col_threshold:
            df = df.filter(col(col_name).isNotNull())  # Drop rows

        else:
            mean_value = df.select(mean(col(col_name))).collect()[0][0]
            df = df.withColumn(col_name, when(col(col_name).isNull(), mean_value).otherwise(col(col_name)))  # Impute with mean

    # Index categorical columns
    for col_name in categorical_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed", handleInvalid="skip")
        df = indexer.fit(df).transform(df)
        df = df.drop(col_name).withColumnRenamed(f"{col_name}_indexed", col_name)

    return df

# Loading and cleaning the data
def load_and_clean_data(file_path: str) -> DataFrame:
    """
    Loads data, cleans it, and returns a cleaned DataFrame.
    """
    if not os.path.exists(parquet_path):
        raw_df = spark.read.csv(file_path, header=True, inferSchema=True)
        raw_df.write.parquet(parquet_path)
    df = spark.read.parquet(parquet_path)
    return clean(df)

# Timing data cleaning process
start_time = time.time()
clean_df = load_and_clean_data(file_path)
clean_df.cache()  # Cache to speed up subsequent operations
cleaning_time = time.time() - start_time
print(f"Data cleaning completed in {cleaning_time:.2f} seconds.")

# Saving the cleaned data
clean_df.write.parquet(cleaned_parquet_path, mode="overwrite")

# Model training with Random Forest Regression
def train_random_forest(df: DataFrame, target_column: str = 'DepDelay') -> tuple:
    """
    Trains a Random Forest model for regression and returns the predictions and training time.
    """
    start_time = time.time()

    # Feature engineering: Assemble features into a single vector
    feature_columns = [col for col in df.columns if col != target_column]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    # Split data into training and testing sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Define the Random Forest Regressor model
    rf = RandomForestRegressor(featuresCol="scaledFeatures", labelCol=target_column)

    # Cross-validation and hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(rf.maxDepth, [5, 10]) \
        .addGrid(rf.numTrees, [20, 50]) \
        .build()

    evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")
    cv = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

    # Train the Random Forest model with cross-validation
    model = cv.fit(train_df)

    # Make predictions on the test set
    predictions = model.transform(test_df)

    # Evaluate model performance
    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mae").evaluate(predictions)

    training_time = time.time() - start_time
    print(f"Model training time: {training_time:.2f} seconds")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")

    return predictions, training_time

def create_spark_session(executor_memory: str, executor_cores: int, executor_instances: int, task_cpus: int, partition_size: int):
    """
    Create a new Spark session with specified configurations.

    Parameters:
    - executor_memory (str): Amount of memory per executor (e.g., "4g").
    - executor_cores (int): Number of cores per executor.
    - executor_instances (int): Number of executors.
    - task_cpus (int): Number of CPUs per task.
    - partition_size (int): Number of shuffle partitions.

    Returns:
    - SparkSession: A new Spark session.
    """
    spark = SparkSession.builder \
        .appName("Spark Performance Experiment") \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.executor.cores", executor_cores) \
        .config("spark.executor.instances", executor_instances) \
        .config("spark.task.cpus", task_cpus) \
        .config("spark.sql.shuffle.partitions", partition_size) \
        .getOrCreate()

    return spark

# Run experiments for different configurations
def experiment_with_configurations(partition_size: int, memory: int, executor_cores: int,
                                   executor_instances: int, task_cpus: int):
    """
    Experiment with different configurations for parallelism, executor resources, and measure the
    time taken for data cleaning and model training.

    Parameters:
    - partition_size (int): Number of shuffle partitions.
    - memory (int): Memory allocated to each executor in GB.
    - executor_cores (int): Number of cores per executor.
    - executor_instances (int): Number of executors.
    - task_cpus (int): Number of CPUs allocated per task.

    Returns:
    - A dictionary with configuration and time taken for cleaning and training.
    """
    # Create a new Spark session with the desired configurations
    spark = create_spark_session(f"{memory}g", executor_cores, executor_instances, task_cpus, partition_size)

    # Log the configurations being used
    print(f"\nTesting with Partition Size: {partition_size}, Memory: {memory}GB, Executor Cores: {executor_cores}, "
          f"Executor Instances: {executor_instances}, Task CPUs: {task_cpus}\n")

    # Load and clean data
    start_time = time.time()
    clean_df = load_and_clean_data(file_path)
    cleaning_time = time.time() - start_time

    # Train the model and evaluate performance
    predictions, training_time = train_random_forest(clean_df)

    # Log the results
    results = {
        "partition_size": partition_size,
        "memory": memory,
        "executor_cores": executor_cores,
        "executor_instances": executor_instances,
        "task_cpus": task_cpus,
        "cleaning_time": cleaning_time,
        "training_time": training_time
    }

    

    return results

# Run experiments with various configurations and save results
results = []
configurations = [
    (300, 8, 1, 4, 1), 
    (300, 8, 2, 4, 2), 
    (300, 8, 3, 4, 3), 
    (300, 8, 4, 4, 4), 
]

# Run experiments for different configurations
for config in configurations:
    partition_size, memory, executor_cores, executor_instances, task_cpus = config
    result = experiment_with_configurations(partition_size, memory, executor_cores, executor_instances, task_cpus)
    results.append(result)

# Convert the list of results into a pandas DataFrame for easy analysis and plotting
df_results = pd.DataFrame(results)

# Convert experiment results into a DataFrame for analysis
df_results.to_csv("experiment_results.csv", index=False)

# Display summary of results
print("Experiment Summary:")
print(df_results.describe())
