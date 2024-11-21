import pandas as pd
import os

""" Filter out duplicate index_text from the dataframe """
def filter(file):
    df = pd.read_csv(file)
    print("Number of rows before dropping duplicates:", len(df))
    
    df.drop_duplicates(subset='index_text', keep='first', inplace=True)
    
    print("Number of rows after dropping duplicates:", len(df))
    
    df = df[['text', 'index_text', 'label']]
    
    df.to_csv(file, index=False)

    return df



def main():
    directory = "/Users/ruipenghan/projects/research/ssbrl/data/paper_data_gt"
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            filter(file_path)

if __name__ == "__main__":
    main()
