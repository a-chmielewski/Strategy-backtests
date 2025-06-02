import os
import pandas as pd

def merge_csv_files(folder_path, output_filename="merged_results.csv"):
    """
    Merges all CSV files in the specified folder into a single CSV file.
    Adds a 'strategy_name' column based on the CSV filename.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
        output_filename (str): The name of the output merged CSV file.
    """
    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter out only the CSV files, excluding the output file itself
    csv_files = [f for f in all_files if f.endswith('.csv') and f != output_filename and f != "mergecsv.py"] # also excluding the script itself
    
    if not csv_files:
        print("No CSV files found to merge.")
        return

    # Initialize an empty list to store DataFrames
    df_list = []

    # Loop through each CSV file and append its data to the list
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            df = pd.read_csv(file_path)
            # Extract strategy name from filename (remove .csv extension)
            strategy_name = os.path.splitext(csv_file)[0]
            # Add strategy_name column to the DataFrame
            df['strategy_name'] = strategy_name
            df_list.append(df)
        except Exception as e:
            print(f"Could not read file {csv_file} due to error: {e}")
            
    if not df_list:
        print("No data to merge after attempting to read CSV files.")
        return

    # Concatenate all DataFrames in the list
    merged_df = pd.concat(df_list, ignore_index=True)
    
    # Define the output file path
    output_file_path = os.path.join(folder_path, output_filename)
    
    # Save the merged DataFrame to a new CSV file
    try:
        merged_df.to_csv(output_file_path, index=False)
        print(f"Successfully merged {len(csv_files)} CSV files into {output_file_path}")
        print(f"Added 'strategy_name' column based on filenames")
    except Exception as e:
        print(f"Could not save merged file due to error: {e}")

if __name__ == "__main__":
    # Define the folder path (current directory where the script is located)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    merge_csv_files(current_folder)
