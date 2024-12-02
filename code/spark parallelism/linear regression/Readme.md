## This contains two files:

data_cleaning_training_multi_machine.py	: Uses parallelism

model_cleaning_training_single_machine.py : Doesnt use parallelism

## In the one without parallelism:
Machine learning workflow is executed in a single-threaded mode, where both data cleaning and k-fold cross-validation are performed sequentially without leveraging parallelism. The data is also loaded in dataframe using pandas without leveraging spark dataframes. During the k-fold cross-validation, each fold is processed sequentially, meaning that the model training and evaluation for one fold is completed before moving on to the next. This approach results in a longer overall training time.


## In the one with paralleliem: 
### Distributed Data Loading:
Data is loaded into Spark as a distributed DataFrame.
The use of .repartition(200) ensures data is split into 200 partitions, allowing parallel processing across cores or nodes.
Optimized Spark Session Configuration:

Configured with spark.executor.cores=4, allowing Spark executors to run 4 tasks in parallel.
Configured with spark.sql.shuffle.partitions=200, optimizing shuffling operations during transformations.
### Parallel Data Cleaning:
Transformations such as imputation and filtering (df.filter(), df.drop()) are performed in parallel across partitions.o	When operations like filtering (filter), aggregation (mean), and transformation (withColumn) are performed, Spark processes the data across partitions in parallel using multiple worker nodes or cores.
String indexing (StringIndexer) is applied in parallel for encoding categorical variables.
### Caching Intermediate Results:
clean_df.cache() ensures the cleaned DataFrame is stored in memory, avoiding redundant computations during further processing.
### Cross-Validation with Parallelism:
Utilized Spark's CrossValidator with parallelism=8, enabling parallel evaluation of hyperparameter combinations across multiple cores.
Each fold and parameter combination is executed simultaneously, significantly reducing training time.
### Efficient Feature Scaling:
The StandardScaler processes features in parallel by partitioning the data, leveraging Spark's distributed architecture.
### Parallel Model Training:
Spark's LinearRegression model is trained using distributed computations across partitions.
Gradient descent and other iterative processes are parallelized over the distributed data.
Converted the data to Parquet format for faster I/O operations.

