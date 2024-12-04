import os
import time
import csv
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, when
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Define data cleaning function
def clean(df):
    drop_threshold = 0.6
    row_drop_threshold = 0.05
    total_rows = df.count()

    numerical_cols = [col_name for col_name, dtype in df.dtypes if dtype in ('int', 'double', 'float')]
    categorical_cols = [col_name for col_name, dtype in df.dtypes if dtype == 'string']

    drop_col_threshold = drop_threshold * total_rows
    row_drop_col_threshold = row_drop_threshold * total_rows

    for col_name in numerical_cols:
        missing_count = df.filter(col(col_name).isNull()).count()
        if missing_count > drop_col_threshold:
            df = df.drop(col_name)
        elif missing_count <= row_drop_col_threshold:
            df = df.filter(col(col_name).isNotNull())
        else:
            mean_value = df.select(mean(col(col_name))).collect()[0][0]
            df = df.withColumn(col_name, when(col(col_name).isNull(), mean_value).otherwise(col(col_name)))

    for col_name in categorical_cols:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed", handleInvalid="skip")
        df = indexer.fit(df).transform(df)
        df = df.drop(col_name).withColumnRenamed(f"{col_name}_indexed", col_name)
    
    return df

# Define model training and evaluation function
def train_and_validate(df, target_column='DepDelay'):
    feature_columns = [col for col in df.columns if col != target_column]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(featuresCol="scaledFeatures", labelCol=target_column, predictionCol="prediction")
    param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]).build()
    evaluator = RegressionEvaluator(labelCol=target_column, predictionCol="prediction", metricName="rmse")

    crossval = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)
    cv_model = crossval.fit(train_df)

    predictions = cv_model.transform(test_df)
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    return rmse

# Define performance analysis function
def analyze_performance(parallelism_levels, file_path, parquet_path, target_column='DepDelay'):
    cleaning_times = []
    training_times = []

    for cpus in parallelism_levels:
        print(f"Testing with {cpus} CPUs...")
        spark = SparkSession.builder \
            .appName(f'project-{cpus}-cpus') \
            .config('spark.executor.memory', '8g') \
            .config('spark.executor.cores', cpus) \
            .config('spark.driver.memory', '8g') \
            .config("spark.executor.instances", 2) \
            .config('spark.sql.shuffle.partitions', '200') \
            .getOrCreate()

        if not os.path.exists(parquet_path):
            df = spark.read.csv(file_path, header=True, inferSchema=True)
            df.write.parquet(parquet_path)
        df = spark.read.parquet(parquet_path)

        # Data Cleaning
        start_time = time.time()
        clean_df = clean(df)
        clean_df.write.parquet("cleaned_data.parquet", mode="overwrite")
        cleaning_time = time.time() - start_time
        cleaning_times.append(cleaning_time)
        print(f"Cleaning time with {cpus} CPUs: {cleaning_time:.2f} seconds")

        # Model Training
        clean_df = spark.read.parquet("cleaned_data.parquet")
        start_time = time.time()
        rmse = train_and_validate(clean_df, target_column)
        training_time = time.time() - start_time
        training_times.append(training_time)
        print(f"Training time with {cpus} CPUs: {training_time:.2f} seconds")
        print(f"RMSE with {cpus} CPUs: {rmse:.2f}")
        
        spark.stop()


        # Write cleaning times to CSV
    cleaning_file_path = "cleaning_times_cpu.csv"
    with open(cleaning_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Cleaning Time (seconds)"])
        for idx, t in enumerate(cleaning_times, start=1):
            writer.writerow([idx, t])
    print(f"Cleaning times successfully written to {cleaning_file_path}")
    
    # Write training times to CSV
    training_file_path = "training_times_cpu.csv"
    with open(training_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Training Time (seconds)"])
        for idx, t in enumerate(training_times, start=1):
            writer.writerow([idx, t])
    print(f"Training times successfully written to {training_file_path}")
        
    # Plot results
    print('Cleaning Times:', cleaning_times)
    print('Training Times:', training_times)
    
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(parallelism_levels, cleaning_times, marker='o', label='Cleaning Time')
    # plt.plot(parallelism_levels, training_times, marker='x', label='Training Time')
    plt.title('Training Time vs Number of Cores (Linear Regression ) Partition Size = 200, Threads = 2')
    plt.xlabel('Number of CPUs')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig("CPU_analysis_cleaning_times.png", dpi=300, bbox_inches="tight")
    plt.show()

# Run performance analysis
parallelism_levels = [1, 2, 3, 4]
file_path = "concatenated_data.csv"
parquet_path = "concatenated_data.parquet"
analyze_performance(parallelism_levels, file_path, parquet_path, target_column='DepDelay')
