import pandas as pd
import numpy as np
import os

def split_csv(file_path):
    df = pd.read_csv(file_path)
    sample_df = df.sample(frac=0.2, random_state=42)

    # Remove the sampled rows from the original dataframe
    # df = df.drop(sample_df.index)
    # df.to_csv(file_path, index=False)

    # Extract the original file name
    file_name = os.path.basename(file_path)
    origin_name = os.path.splitext(file_name)[0]
    new_file_name = f"../data/paper_data_gt/{origin_name}_gt.csv"

    # Save the modified dataframe as a new CSV file
    sample_df.to_csv(new_file_name, index=False)


def main():
    directory = "/Users/ruipenghan/projects/research/ssbrl/data/paper_data"
    csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]

    # Loop through each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        split_csv(file_path)


if __name__ == "__main__":
    main()