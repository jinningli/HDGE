import pandas as pd
import os

import glob
csv_files = sorted(glob.glob("./*.csv"))
dataframes = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file, index_col=0)
    df = df[["message_id", "actor_id", "engagement_type", "engagement_parent", "text", "time_published", "index_text", "tweet_counts", "user_counts", "gpt_label", "manual_label", "is_gt"]]
    df["belief"] = csv_file.split("/")[-1].replace(".csv", "")
    dataframes.append(df)
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_csv("combined_data.csv", index=False)
print(combined_df)