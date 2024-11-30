#MERGING CSV FILES 

# # Define the folder path where your CSV files are stored
folder_path = "C:/Users/mruna/Documents/spark/project/dataverse_files/data"

# Initialize an empty list to store DataFrames
dataframes = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Check if the file is a CSV
        file_path = os.path.join(folder_path, filename)
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
final_dataframe = pd.concat(dataframes, ignore_index=True)

# Determine the number of records (rows)
num_records = len(final_dataframe)
# Output the results
print(f'Number of records: {num_records}')
# Optionally, save the concatenated DataFrame to a new CSV
final_dataframe.to_csv("concatenated_data.csv", index=False)
