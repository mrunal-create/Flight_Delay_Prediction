## Performance Analysis: 


## Time Taken without parallelism :
The presented output demonstrates a machine learning workflow executed in a single-threaded mode, where both data cleaning and k-fold cross-validation are performed sequentially without leveraging parallelism. The data cleaning phase, completed in 39.22 seconds, but it did not utilize distributed or parallel processing frameworks, such as Spark's distributed computing capabilities or multi-threading.

Similarly, during the k-fold cross-validation, each fold was processed sequentially, meaning that the model training and evaluation for one fold were completed before moving on to the next. This approach results in a longer overall training time, as evidenced by the total training duration of 157.01 seconds.


## Time taken with parallelism using multi machine (CPU) and spark:
With parallelism enabled:

Data Cleaning Time: Reduced to 34.25 seconds compared to longer durations without parallelism. This improvement is attributed to the distributed processing capabilities of Spark, where data is partitioned and processed across multiple CPU cores concurrently.

Model Training Time: Reduced to 36.14 seconds, leveraging the parallelism settings in Spark and the CrossValidator with increased parallelism. This parallel execution of multiple model parameter combinations and faster computations in the pipeline significantly reduced the overall runtime.


