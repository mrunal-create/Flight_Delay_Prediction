


import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Initialize Spark session with optimized configuration
spark = SparkSession.builder \
    .appName('Optimized Cleaning and Training') \
    .config('spark.executor.memory', '8g') \
    .config('spark.executor.cores', '4') \
    .config('spark.driver.memory', '8g') \
    .config('spark.sql.shuffle.partitions', '200') \
    .getOrCreate()

# File paths
file_path = "concatenated_data.csv"
parquet_path = "concatenated_data.parquet"
cleaned_parquet_path = "cleaned_data.parquet"

# Data cleaning function
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

# Load and clean data
if not os.path.exists(parquet_path):
    raw_df = spark.read.csv(file_path, header=True, inferSchema=True)
    raw_df.write.parquet(parquet_path)
df = spark.read.parquet(parquet_path)



# Start timing for data cleaning
cleaning_start_time = time.time()
clean_df = clean(df)
clean_df.cache()  # Cache the cleaned DataFrame to avoid recomputation
cleaning_time = time.time() - cleaning_start_time


# Save the cleaned DataFrame
clean_df.write.parquet(cleaned_parquet_path, mode="overwrite")

# Model training function
def train_and_predict(df, target_column='DepDelay'):
    """
    Trains a Linear Regression model with cross-validation to predict the target column.
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

    # Train Linear Regression model with cross-validation
    lr = LinearRegression(featuresCol="scaledFeatures", labelCol=target_column, predictionCol="prediction")

    # Cross-validation setup
    param_grid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
        .build()

    evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")
    cv = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, parallelism=8)

    # Train the model
    cv_model = cv.fit(train_df)

    # Make predictions on the test set
    predictions = cv_model.transform(test_df)

    # Evaluate the model
    rmse = evaluator.evaluate(predictions)
    mae = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mae").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2").evaluate(predictions)
    

    # End timing
    training_time = time.time() - start_time
   

    return predictions, training_time,rmse,mae,r2

# Train the model
predictions, training_time,rmse,mae,r2 = train_and_predict(clean_df, target_column='DepDelay')
print('RMSE',rmse)
print('R2',r2)
print('MAE',mae)
# Stop the Spark session
spark.stop()

# Performance Summary


print("Performance Summary:- Data Cleaning Time ", cleaning_time, "Model Training Time",  training_time)

