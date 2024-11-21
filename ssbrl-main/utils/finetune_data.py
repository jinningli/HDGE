import pandas as pd
import os

def filter_and_sample_csv(csv_path):
    # Read the CSV file as a pandas dataframe
    df = pd.read_csv(csv_path)
    
    # Filter out rows with is_gt = 1
    df = df[df['is_gt'] == 1]
    
    # Drop duplicates by the field "index_text"
    df = df.drop_duplicates(subset='index_text', keep="first")
    
    # Randomly select 20% of the remaining rows
    df_sampled = df.sample(frac=0.2)
    
    # Save the sampled dataframe to a new CSV file
    file_name = os.path.basename(csv_path)
    output_path = os.path.join("ft_data/gt_data", file_name.replace('.csv', '_gt_ft_data.csv'))
    df_sampled.to_csv(output_path, index=False)


def process_directory(directory):
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Filter out non-CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    
    # Process each CSV file
    for file in csv_files:
        file_path = os.path.join(directory, file)
        filter_and_sample_csv(file_path)


def merge_all_ft_data(directory):
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Filter out non-CSV files and exclude "all_finetune_data.csv"
    csv_files = [file for file in files if file.endswith('.csv') and file != 'all_finetune_data.csv']
    
    # Initialize an empty dataframe to store the merged data
    merged_df = pd.DataFrame()
    
    # Merge all CSV files
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    # Save the merged dataframe to a new CSV file
    output_path = os.path.join(directory, 'all_finetune_data.csv')
    merged_df.to_csv(output_path, index=False, header=True)


def main():
    output_dir = "ft_data/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_directory('/Users/ruipenghan/projects/research/ssbrl/data/paper_data_merged')

    merge_all_ft_data("/Users/ruipenghan/projects/research/ssbrl/utils/ft_data/gt_data")
    # filter_and_sample_csv("/Users/ruipenghan/projects/research/ssbrl/data/paper_data_merged/filtered_data_10_5_5_20000_tree_vis_us_military-philippine_labeled_merged.csv")

if __name__ == "__main__":
    main()