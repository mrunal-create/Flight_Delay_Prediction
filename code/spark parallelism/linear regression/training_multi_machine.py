import os
import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator




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
    #print(f"Root Mean Squared Error (RMSE) on test data: {rmse:.2f}")
    mae = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2").evaluate(predictions)
    print(f"Validation Results: RMSE = {rmse:.2f}, MAE = {mae:.2f}, R² = {r2:.2f}")

    # End timing
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    return predictions, end_time - start_time


# Fixed Configuration
print("Testing with fixed configuration: 4 CPUs and 4g memory...")

# Initialize SparkSession with a fixed configuration
spark = SparkSession.builder \
    .appName('project-4-cpus-4g') \
    .config('spark.executor.memory', '8g') \
    .config('spark.executor.cores', 8) \
    .config('spark.driver.memory', '8g') \
    .config('spark.sql.shuffle.partitions', '200') \
    .getOrCreate()

# Load the cleaned data
df = spark.read.parquet("cleaned_data.parquet")

# Train the model and measure training time
predictions, time_taken = train_and_predict(df, target_column='DepDelay')

# Print training time
print(f"\nTraining completed with 4 CPUs and 4g memory in {time_taken:.2f} seconds.")

# Stop Spark session
spark.stop()
