import argparse
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
import pandas as pd
from scipy.special import softmax
import os
from sklearn.cluster import KMeans
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def get_cls_embedding(tweet):
    global model, tokenizer
    #model.save_pretrained(MODEL)
    # text = "Covid cases are increasing fast! Covid cases are increasing fast! "
    # text = preprocess(text)
    encoded_input = tokenizer(tweet, return_tensors='pt')

    output = model(**encoded_input, output_hidden_states=True)
    # print("output:")
    # print(len(output.hidden_states))
    # print(output.hidden_states[-1])

    last_layer_hidden_states = output.hidden_states[-1]
    # Option 1: Use the embedding of the [CLS] token (index 0)
    cls_embedding = last_layer_hidden_states[:, 0, :]
    # print(cls_embedding.shape)
    # print(cls_embedding)
    return cls_embedding.detach().numpy()


def run(args):
    csv_path = args.data
    data = pd.read_csv(csv_path)
    data = data.drop_duplicates(subset="index_text", keep="first")
    data = data.drop_duplicates(subset="text", keep="first")
    tweets = data["text"].tolist()
    index_text = data["index_text"].tolist()
    assert(len(tweets) == len(index_text))
    embeddings = []
    start_time = time.time()
    # i = 0
    for tweet in tweets:
        # if i == 5:
            # break
        embeddings.append(get_cls_embedding(tweet))
        # i += 1

    # Convert the list of embeddings to a numpy array
    embeddings_array = np.array(embeddings)

    # Reshape (5, 1, 768) to (5, 768)
    embeddings_array = np.squeeze(embeddings_array)

    # Create an instance of KMeans with 2 clusters
    kmeans = KMeans(n_clusters=2)

    # Fit the KMeans model to the embeddings
    kmeans.fit(embeddings_array)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time to run roberta-base kmeans on topic {args.topic}: {total_time}")
    # exit(1)
    # Get the cluster labels for each embedding
    cluster_labels = kmeans.labels_

    clustered_data = pd.read_csv(csv_path)
    for i, label in enumerate(cluster_labels):
        km_label = "opposing"
        if label == 1:
            km_label = "supportive"
        idx_text = index_text[i]
        # print((clustered_data['index_text'] == idx_text).sum())
        # exit(1)
        clustered_data.loc[clustered_data['index_text'] == idx_text, 'kmean_clsuter'] = km_label

    folder_path = f"labelled/"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Define the file path
    file_path = os.path.join(folder_path, f"{args.topic}_kmean_clustered.csv")

    # Save the dataframe as a CSV file
    clustered_data.to_csv(file_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--data", required=True, type=str, help="Raw file to be clustered")
    parser.add_argument("--topic", required=True, type=str, help="Topic of the data")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()