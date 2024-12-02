import os
import time
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# File path
file_path = "concatenated_data.csv"

# Data Cleaning Function
def clean(df):
    """
    Cleans the input DataFrame by:
    - Imputing missing values in numerical columns with the mean.
    - Dropping numerical columns with more than 40% missing values.
    - Encoding categorical columns using label encoding.
    """
    # Drop columns with more than 40% missing values
    drop_threshold = 0.6  # 60% missing threshold for dropping columns
    row_count = len(df)
    df = df.dropna(axis=1, thresh=int(drop_threshold * row_count))
    
    # Impute missing numerical values with mean
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Label encode categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Optimize memory usage by converting numerical columns to float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    return df

# Load Data
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found!")

# Read the CSV into a Pandas DataFrame
df = pd.read_csv(file_path)

# Start timing
start_time = time.time()

# Clean the DataFrame
clean_df = clean(df)
cleaning_time = time.time() - start_time
print(f"Data cleaning completed in single-threaded mode in {cleaning_time:.2f} seconds.")

# Model Training with Cross-Validation
def train_and_cross_validate(df, target_column='DepDelay', n_splits=5):
    """
    Trains a Linear Regression model using k-fold cross-validation and evaluates performance.
    """
    start_time = time.time()

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1

    # Initialize lists to store validation metrics
    rmse_list = []

    # Perform cross-validation
    for train_index, test_index in kf.split(X):
        print(f"Training fold {fold}/{n_splits}...")
        # Split into train and test sets for this fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Linear Regression model
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)

        # Make predictions on the test set
        y_pred = lr.predict(X_test_scaled)

        # Evaluate the model
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse_list.append(rmse)

        print(f"Fold {fold} - RMSE: {rmse:.2f}")
        fold += 1

    # Calculate average metrics across all folds
    avg_rmse = sum(rmse_list) / n_splits

    print("\nCross-Validation Results:")
    print(f"Average RMSE: {avg_rmse:.2f}")

    # End timing
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    return avg_rmse

# Perform training with cross-validation
avg_rmse = train_and_cross_validate(clean_df)
