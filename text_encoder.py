import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Load your data
data = pd.read_csv('datasets/combined_data.csv')
# https://huggingface.co/eventdata-utd/conflibert-insight-crime-multilabel
# Initialize tokenizer and model from Hugging Face
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("snowood1/ConfliBERT-scr-uncased")
model = AutoModel.from_pretrained("snowood1/ConfliBERT-scr-uncased")
# tokenizer = AutoTokenizer.from_pretrained("eventdata-utd/conflibert-insight-crime-multilabel")
# model = AutoModel.from_pretrained("eventdata-utd/conflibert-insight-crime-multilabel")
# tokenizer = AutoTokenizer.from_pretrained("mlburnham/Political_DEBATE_large_v1.0")
# model = AutoModel.from_pretrained("mlburnham/Political_DEBATE_large_v1.0")
# https://huggingface.co/mlburnham/Political_DEBATE_large_v1.0

model.eval()

# Function to encode texts to embeddings using the model (now handles batches)
def encode_texts(texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Extract the CLS token's embedding (first token of the last hidden state)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(batch_embeddings)

        # # Calculate the mean of the last hidden state across all tokens
        # batch_embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        # embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Deduplicate texts
unique_texts = data['text'].drop_duplicates().tolist()
unique_embeddings = encode_texts(unique_texts)

# Map embeddings back to all texts
embedding_dict = dict(zip(unique_texts, unique_embeddings))
data['embeddings'] = data['text'].map(embedding_dict)

data = data.drop_duplicates(subset="text", keep="first")

# Perform clustering
n_clusters = 7  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
data['cluster'] = kmeans.fit_predict(data['embeddings'].tolist())

# Evaluation metrics
labels_true = data['topic']
labels_pred = data['cluster']
ari = adjusted_rand_score(labels_true, labels_pred)
nmi = normalized_mutual_info_score(labels_true, labels_pred)
silhouette = silhouette_score(data['embeddings'].tolist(), labels_pred)
# Print the evaluation results
print("Adjusted Rand Index (ARI):", ari)
print("Normalized Mutual Information (NMI):", nmi)
print("Silhouette Score:", silhouette)

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(data['embeddings'].tolist())

# Plotting the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=data['cluster'], cmap='viridis')
plt.colorbar(scatter)
plt.title('Text Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig("text_encoder.png")

# Save the clustering results to a new CSV file
data = data.drop(columns=["embeddings"])
data = data.sort_values(["topic"])
data.to_csv('clustering_results.csv', index=False)
