import argparse
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy.special import softmax
import os
from sklearn.cluster import KMeans
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

access_token = os.environ.get("HUGGING_FACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("Ruipeng/tweet-roberta-full-ep300", token=access_token)
model = AutoModelForSequenceClassification.from_pretrained("Ruipeng/tweet-roberta-full-ep300", token=access_token)

config = AutoConfig.from_pretrained("Ruipeng/tweet-roberta-full-ep300")


def inference(tweet):
    global model, tokenizer, config
    # tweet = "RT @BenjaminNorton Britain and Japan (both of which previously colonized parts of China) are joining the US in preparing for war with China. Britain and Japan signed a military agreement allowing troops to be deployed to each other's country, while planning joint exercises https://t.co/FlBnACVPoq"
    encoded_input = tokenizer(tweet, return_tensors='pt')

    output = model(**encoded_input, output_hidden_states=True)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    # res = {}
    # for i in range(scores.shape[0]):
        # l = config.id2label[ranking[i]]
        # s = scores[ranking[i]]
        # print(f"{i+1}) {l} {np.round(float(s), 4)}")
        # res[l] = s

    # Get the label with the highest score
    return config.id2label[ranking[0]]


def run(args):
    csv_path = args.data

    # Prepare the test data
    data = pd.read_csv(csv_path)
    data = data[data['is_gt'] == 1]
    data = data.drop_duplicates(subset='index_text', keep="first")

    result_mapping = {}
    # Iterate over each row of data
    for index, row in data.iterrows():
        text = row["text"]
        label = inference(text)
        result_mapping[row["index_text"]] = label


    result_csv = pd.read_csv(csv_path)

    for index_text, label in result_mapping.items():
        result_csv.loc[result_csv['index_text'] == index_text, 'gpt_label'] = label

    folder_path = f"labeled_data/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    # Define the file path
    file_path = os.path.join(folder_path, f"{args['topic']}_twhinbert_ft_res.csv")

    # Save the dataframe as a CSV file
    result_csv.to_csv(file_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--data", required=True, type=str, help="Raw file to be inferenced by the finetuned model")
    parser.add_argument("--topic", required=True, type=str, help="Topic of the data")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
