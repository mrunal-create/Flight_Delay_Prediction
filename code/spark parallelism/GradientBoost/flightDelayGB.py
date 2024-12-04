import os
import time
import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import GBTRegressor
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, exp



spark = SparkSession.builder \
    .appName('Optimized Cleaning and Training') \
    .config('spark.executor.instances', '3') \
    .config('spark.executor.memory', '12g') \
    .config('spark.executor.cores', '5') \
    .config('spark.driver.memory', '12g') \
    .config('spark.sql.shuffle.partitions', '50') \
    .getOrCreate()


# cleaning_start_time = time.time()
# df = spark.read.parquet('/home/ghosh.shu/EECE/flightData.parquet')
# print("Data Loaded-------------------------------")


def iqr_outlier_treatment(dataframe, columns, factor=1.5):
    """
    Detects and treats outliers using IQR for multiple variables in a PySpark DataFrame.

    :param dataframe: The input PySpark DataFrame
    :param columns: A list of columns to apply IQR outlier treatment
    :param factor: The IQR factor to use for detecting outliers (default is 1.5)
    :return: The processed DataFrame with outliers treated
    """
    for column in columns:
        # Calculate Q1, Q3, and IQR
        quantiles = dataframe.approxQuantile(column, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        # Define the upper and lower bounds for outliers
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Filter outliers and update the DataFrame
        dataframe = dataframe.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))

    return dataframe

def clean_data(df):
    df = iqr_outlier_treatment(df, ['DepDelay'], factor=3)

    categorical_cols = ['UniqueCarrier', 'Origin', 'Dest',
                        'Year', 'Month' , 'DayofMonth' , 'DayOfWeek']

    numerical_cols = ['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime', 'ActualElapsedTime',
                      'CRSElapsedTime', 'AirTime', 'ArrDelay','DepDelay', 'Distance', 'LateAircraftDelay']

    df = df.select(categorical_cols + numerical_cols)

    num_rows = df.count()

    for col_name in numerical_cols:
        nanCount = df.filter(col(col_name).isNull()).count()
        print(f"Column '{col_name}' has {nanCount} missing values.")
        if (nanCount/num_rows) > 0.4:
            df = df.drop(col_name)
            print(f"Column '{col_name}' has more than 40% missing values. Dropped.")
        elif(nanCount/num_rows) > 0.1:
            mean_value = df.select(mean(col(col_name))).collect()[0][0]
            print(f"Column '{col_name}' has {nanCount} missing values. Imputing with mean: {mean_value}")
            df = df.withColumn(col_name, when(col(col_name).isNull(), mean_value).otherwise(col(col_name)))
        else:
            print(f"Column '{col_name}' has {nanCount} missing values. No action taken.")
            continue

    df = df.na.drop(how="any")

    num_rows = df.count()

    print("Number of rows after preproc : ",num_rows)

    for col_name in categorical_cols:
      indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed", handleInvalid="skip")
      df = indexer.fit(df).transform(df)
      df = df.drop(col_name).withColumnRenamed(f"{col_name}_indexed", col_name)
      print(f"Column '{col_name}' has been indexed.")

    return df

def train_and_validate(df, target_column='DepDelay'):
    """
    Trains a Linear Regression model with cross-validation to predict the target column.
    """
    # start_time = time.time()

    # Select features (all columns except the target)
    feature_columns = [col for col in df.columns if col != target_column]

    print(f"Selected features: {feature_columns}")

    # Assemble features into a single vector
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    print("Feature vector assembled.")

    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    print("Features scaled.")

    # Split data into train and test sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    gbt = GBTRegressor(featuresCol="scaledFeatures", labelCol=target_column, predictionCol="prediction")

    # Cross-validation setup
    param_grid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [2]) \
        .build()

    evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=2, parallelism=8)

    print("Model training started...")

    # Train the model
    cv_model = cv.fit(train_df)

    print("Model training completed.")

    # End timing
    # training_time = time.time() - start_time
    # print(f"Total training time: {training_time:.2f} seconds.")

    # Make predictions on the test set
    predictions = cv_model.transform(test_df)

    print("Predictions made.")

    # Evaluate the model
    rmse = evaluator.evaluate(predictions)
    # mae = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="mae").evaluate(predictions)
    # r2 = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="r2").evaluate(predictions)
    print(f"Validation Results: RMSE = {rmse:.2f}")



    return rmse


parquet_path = "/home/ghosh.shu/EECE/flightData.parquet"
df = spark.read.parquet(parquet_path)

# Data Cleaning
start_time = time.time()
clean_df = clean_data(df)
# clean_df.write.parquet("cleaned_data.parquet", mode="overwrite")
cleaning_time = time.time() - start_time
print(f"Cleaning time : {cleaning_time:.2f} seconds")

start_time = time.time()
rmse = train_and_validate(clean_df, target_column)
training_time = time.time() - start_time
training_times.append(training_time)
print(f"Training time : {training_time:.2f} seconds")
print(f"RMSE : {rmse:.2f}")

spark.stop()
