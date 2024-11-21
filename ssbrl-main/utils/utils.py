import os
import pandas as pd
import random
import glob
def unlabel_data(source_dir, target_dir):

    # Get all CSV files in the source directory
    csv_files = [file for file in os.listdir(source_dir) if file.endswith('.csv')]

    for file in csv_files:
        # Read the CSV file
        file_path = os.path.join(source_dir, file)
        df = pd.read_csv(file_path)
        df.rename(columns={'Unnamed: 0':''}, inplace=True)

        # Remove the "label" column
        if 'label' in df.columns:
            df.drop('label', axis=1, inplace=True)

        # Save the modified DataFrame to the target directory
        target_file_path = os.path.join(target_dir, file.replace('_labeled.csv', '.csv'))
        df.to_csv(target_file_path, index=False)


def count_unique_idx_text(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Count the unique number of index_text
    unique_count = df['index_text'].nunique()
    print(unique_count)
    return unique_count

def sample_from_gt(merged_file):
    # Read the CSV file
    df = pd.read_csv(merged_file)

    # Get the unique index_text with is_gt = 1
    unique_gt_index_text = df.loc[df['is_gt'] == 1, 'index_text'].unique()

    # Randomly select 30% of the unique index_text
    sample_size = int(len(unique_gt_index_text) * 0.6)
    sampled_index_text = random.sample(list(unique_gt_index_text), sample_size)

    # Assign is_gt = 0 and set "manual_label" column to empty for the sampled index_text
    df.loc[df['index_text'].isin(sampled_index_text), 'is_gt'] = 0
    df.loc[df['index_text'].isin(sampled_index_text), 'manual_label'] = ''
    df.rename(columns={'Unnamed: 0':''}, inplace=True)

    # Save the modified DataFrame back to the CSV file
    df.to_csv(merged_file, index=False)


def count_neutral(dir, output_file, model):
    # Get all CSV files in the directory
    # csv_files = [file for file in os.listdir(dir) if file.endswith('.csv')]
    csv_files = glob.glob(os.path.join(dir, '**/*.csv'), recursive=True)
    with open(output_file, 'a') as f:
        f.write("=" * 20 + f"For {model}" + "=" * 20 + '\n')

    for file in csv_files:
        # Read the CSV file
        file_path = os.path.join(dir, file)
        df = pd.read_csv(file_path)
        df = df[df['is_gt'] == 1]
        df = df.drop_duplicates(subset="index_text", keep="first")
        count = df[df['gpt_label'] == 'neutral'].shape[0]
        assert count == len(df[df['gpt_label'] == 'neutral'])
        # Write the count to the output file
        with open(output_file, 'a') as f:
            # Get the file name from the full path
            file_name = os.path.basename(file)
            f.write(f"Neutral Count of {file_name}: {str(count)}\n")


def main():
    source_dir = "/Users/ruipenghan/projects/research/ssbrl/data/paper_data"
    target_dir = "/Users/ruipenghan/projects/research/ssbrl/data/paper_data_unlabeled"
    # Call the function with the source and target directories
    unlabel_data(source_dir, target_dir)

if __name__ == "__main__":
    # count_unique_idx_text("/Users/ruipenghan/projects/research/ssbrl/data/paper_data/filtered_data_10_5_5_20000_tree_vis_us_military-nato_labeled.csv")

    # sample_from_gt("/Users/ruipenghan/projects/research/ssbrl/data/paper_data_merged/filtered_data_10_5_5_20000_tree_vis_labor_and_migration-china_labeled_merged.csv")

    count_neutral("/Users/ruipenghan/projects/research/ssbrl/data/paper_data_merged", "neutral_count.txt", "GPT3")
    count_neutral("/Users/ruipenghan/projects/research/ssbrl/baselines/gpt-4/labeled_data/first_run_labels", "neutral_count.txt", "GPT4")