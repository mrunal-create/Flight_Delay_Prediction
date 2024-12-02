import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize SparkSession in local mode
spark = SparkSession.builder \
    .appName('SingleMachineDataProcessing') \
    .master("local[*]").config('spark.executor.memory', '4g').config('spark.driver.memory', '4g').config('spark.sql.shuffle.partitions', '1').getOrCreate()

# File path
file_path = "concatenated_data.csv"
parquet_path = "data_single_machine.parquet"

# Data Cleaning Function
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

# Load Data
if not os.path.exists(parquet_path):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    df.write.parquet(parquet_path)
df = spark.read.parquet(parquet_path)

# Data Cleaning
start_time = time.time()
clean_df = clean(df)
cleaning_time = time.time() - start_time
print(f"Data cleaning completed in single-machine mode in {cleaning_time:.2f} seconds.")

# Model Training
def train_and_predict(df, target_column='DepDelay'):
    """
    Trains a Linear Regression model to predict the target column using PySpark.
    """
    start_time = time.time()

    # Select features (all columns except the target)
    feature_columns = [col for col in df.columns if col != target_column]
    
    # Assemble features into a single vector
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # Split data into train and test sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Train Linear Regression model
    lr = LinearRegression(featuresCol="scaledFeatures", labelCol=target_column, predictionCol="prediction")
    lr_model = lr.fit(train_df)

    # Make predictions on the test set
    predictions = lr_model.transform(test_df)

    # Evaluate the model
    evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) on test data: {rmse:.2f}")

    # End timing
    end_time = time.time()
    # print(f"Model training completed in single-machine mode in {end_time - start_time:.2f} seconds.")
    
    return predictions

# Train the model
predictions = train_and_predict(clean_df)
training_time = time.time() - start_time
print(f"Model training completed in {training_time:.2f} seconds.")

# Validation Analysis
evaluator = RegressionEvaluator(labelCol="DepDelay", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Validation RMSE: {rmse:.2f}")

# Stop SparkSession
spark.stop()
