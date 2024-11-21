import pandas as pd

def split_task(parquet_file):
    # Read Parquet file as CSV
    df = pd.read_parquet(parquet_file)

    # Drop duplicates based on "index_text" column
    df.drop_duplicates(subset='index_text', keep='first', inplace=True)
    
    # Split the DataFrame into three parts
    split_index1 = len(df) // 3
    split_index2 = 2 * len(df) // 3
    df1 = df.iloc[:split_index1]
    df2 = df.iloc[split_index1:split_index2]
    df3 = df.iloc[split_index2:]
    
    # Save the split DataFrames as CSV files
    df1.to_csv('first_third.csv', index=False)
    df2.to_csv('second_third.csv', index=False)
    df3.to_csv('last_third.csv', index=False)

def main():
    parquet_file = '/path/to/parquet_file.parquet'
    split_task(parquet_file)

if __name__ == "__main__":
    main()