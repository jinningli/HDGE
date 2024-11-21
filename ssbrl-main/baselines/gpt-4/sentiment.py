import argparse

import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
# import utils
import datetime
import json
from tqdm import tqdm
# Load environment variables

load_dotenv()


# Chat API Docs: https://platform.openai.com/docs/api-reference/chat
def chat(client = None, user_prompt = "", sys_prompt = "", model="gpt-4-1106-preview", len_limit=4000, temperature=0.5):
    # print(f"model is {model}")
   # Creating a message as required by the API
    messages = [
        # Defining system role and content
        {"role": "system", "content": sys_prompt},

        # Defining the actual prompt
        {"role": "user", "content": user_prompt}
    ]
  
   # Calling the ChatCompletion API
    response = client.chat.completions.create(
       model=model,
       messages=messages,
       temperature=temperature, # Between 0-2, defines the randomness of the response, 0 as deterministic, 2 as very random
       max_tokens=len_limit, # Maximal length of the response
       response_format={"type": "json_object"},
   )

    # Response object API: https://platform.openai.com/docs/api-reference/chat/object
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description='This program requires you to have all pickle files stored in a directory. The following arguments are optional.')

    parser.add_argument('--topic', type=str, required=True, help='The current topic')
    
    parser.add_argument('--data', type=str, required=True, default='', help='the path to the raw csv file containing the tweets')

    parser.add_argument('--key', type=str, required=True, default='', help='RUIJIE_KEY, WALLY_KEY, or OPENAI_API_KEY')

    parser.add_argument('--result_mapping', type=str, required=True, default="result_mapping.json", help='Result is saved to this file')

    parser.add_argument('--dest_folder', type=str, required=True, help='destination foder for saving the labeled file.')

    parser.add_argument('--instruction', type=str, required=True, help='instruction for the prompt to GPT model. Default is prompts/edca_prompt.txt')

    parser.add_argument('--word_limit', type=int, default=4000, help='the word limit for the response from GPT model. Default is 300.')

    parser.add_argument('--model', type=str, default='gpt-4-1106-preview', help='the model to use for the chatbot. Default is gpt-4-1106-preview; available: gpt-3.5-turbo-1106')

    parser.add_argument('--batch', type=int, default=10, help='batch size for the tweets to be put into the prompt. Default is 30.')

    parser.add_argument('--temperature', type=int, default=0.5, help='defines the randomness of the response, 0 as deterministic, 2 as very random. Default is 0.5')
    args = parser.parse_args()

    # Read prompt instruction from instruction.txt
    try:
        with open(args.instruction, "r") as ins_file:
            instruction = ins_file.read()
    except FileNotFoundError:
        print(f"Instruction file {args.instruction} not found. Exiting...")
        exit(1)

    if args.key not in ["RUIJIE_KEY" , "WALLY_KEY", "RAY_KEY", "OPENAI_API_KEY"]:
        print(f"--key args must be in [RUIJIE_KEY , WALLY_KEY, RAY_KEY, OPENAI_API_KEY]")
        exit(1)
    
    client = OpenAI(
        api_key= os.environ.get(args.key),
    )

    word_limit = args.word_limit
    temperature = args.temperature

    # Log preparation
    if not os.path.exists(f"analysis/{args.topic}"):
        os.makedirs(f"analysis/{args.topic}")
    log_file = os.path.join(f"analysis/{args.topic}", f"{args.topic}_analysis_seq={str(datetime.datetime.now())[-5:]}.log")

    # Create dest folder
    if not os.path.exists(args.dest_folder + "/" + args.topic):
        os.makedirs(args.dest_folder + "/" + args.topic)

    # Read CSV as pandas dataframe
    batch_size = args.batch
    result_mapping = {}  # mapping from index_text to label
    pos, neg, neu = 0, 0, 0
    # Prepare CSV file: initialize lable column or select rows with null labels.
    csv_file = args.data
    all_rows = pd.read_csv(csv_file)

    # if 'label' not in all_rows.columns:
    #     all_rows['label'] = ''  
    # else:
    #     all_rows = all_rows[all_rows['label'].isnull()]

    # Filter out rows with is_gt = 0
    all_rows = all_rows[all_rows['is_gt'] == 1]
    print(f"Total raw tweets: {len(all_rows)}")
    all_rows.drop_duplicates(subset="index_text", keep="first", inplace=True)
    print(f"Unique tweets to be processed: {len(all_rows)}")
    # Sample data if necessary
    # all_rows = all_rows.sample(frac=0.3, random_state=42) 
    # print(f"After random sampling down to 30%: {len(all_rows)}")
    for i in tqdm(range(0, len(all_rows), batch_size), total=len(list(range(0, len(all_rows), batch_size)))):
        if i + batch_size > len(all_rows):
            rows = all_rows.iloc[i:]
        else:
            rows = all_rows.iloc[i:i + batch_size]

        prompt = instruction
        prompt += "\nHere are the following tweets (Tweet ID: Tweet) for you to analyze its sentiment on US military, separated by lines:\n"

        for _, row in rows.iterrows():
            id_, tweet = row['message_id'], row['text']
            prompt += f"Tweet {id_}: {tweet}\n\n"

        # print(prompt)
        # Feed to GPT model
        try:
            completion = chat(client=client, user_prompt=prompt, sys_prompt="", model=args.model, len_limit=word_limit, temperature=temperature)
        except Exception as e:
            print(e)
            # print(prompt)
            print(f"Error occurred while processing batch {i}. Skipping to next batch...")
            continue
        # print(completion)
        # exit(1)
        completion_json = json.loads(completion)
        # Retrive list of tweet ids and their sentiments
        sentiments = completion_json["tweets"]
        for tweet in sentiments:
            id = tweet["ID"]
            sentiment = tweet["Sentiment"]
            tweet_id = type(all_rows['message_id'].iloc[0])(id)

            # Check if there are any matching rows
            matching_rows = all_rows[all_rows['message_id'] == tweet_id]['index_text']
            if not matching_rows.empty:
                result_mapping[matching_rows.iloc[0]] = sentiment
                if sentiment == "supportive":
                    pos += 1
                elif sentiment == "opposing":
                    neg += 1
                else:
                    neu += 1
            else:
                # If the ID returned from GPT is invalid, skip
                pass

        # Append completion to completion.log file
        with open(log_file, "a") as f:
            f.write(completion + "\n\n")

    print("Done labeling. Saving results...")
    result_file = args.result_mapping
    with open(result_file, "w", encoding="utf-8") as fout:
        json.dump(result_mapping, fout, indent=2, ensure_ascii=False)

    # Label the csv file and save to new files.
    data = pd.read_csv(csv_file)
    data.rename(columns={'Unnamed: 0':''}, inplace=True)

    # Write label to data that is in result_mapping; regardless of its original label (to ensure consistency)
    for index_text, sentiment in result_mapping.items():
        data.loc[data["index_text"] == index_text, "gpt_label"] = sentiment
        # data["label"][data["index_text"] == index_text] = sentiment

    # Saving labeled file to the dest directory
    labeled_csv_file = os.path.basename(csv_file).replace(".csv", "_labeled.csv")
    labeled_parquet_file = os.path.basename(csv_file).replace(".csv", "_labeled.parquet")
    dest_folder = args.dest_folder
    # os.makedirs(dest_folder, exist_ok=True)  # Create the destination folder if it doesn't exist

    data.to_csv(f"{dest_folder}/{args.topic}/{labeled_csv_file}", index=False)
    data.to_parquet(f"{dest_folder}/{args.topic}/{labeled_parquet_file}")

    with open(log_file, "a") as f:
        f.write(f"Statistics: supportive: {pos}, opposing: {neg}, neutral: {neu}\n")

    print(f"Analysis Completed, saved all logs to {log_file}")

if __name__ == "__main__":
    main()

