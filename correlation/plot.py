import pickle
import numpy as np
import json

dirs = [
    "results/belief_embedding_output_Crime_2_0.25_0.25",
    "results/belief_embedding_output_EDCA_2_0.25_0.25",
    "results/belief_embedding_output_Energy_Issues_China_2_0.25_0.25",
    "results/belief_embedding_output_Labor_and_Migration_China_2_0.25_0.25",
    "results/belief_embedding_output_Social_and_Economic_Issues_Philippines_2_0.05_0.05",
    "results/belief_embedding_output_United_States_Military_Philippine_2_0.25_0.25"
]

pro_map = [
    "Pro-Government (Crime)",
    "Pro-EDCA (EDCA)",
    "Pro-China (Energy China)",
    "Pro-China (Labor Migration China)",
    "Pro-Philippines (Social Economics)",
    "Pro-Western (US Military Philippines)",
]

anti_map = [
    "Anti-Government (Crime)",
    "Anti-EDCA (EDCA)",
    "Anti-China (Energy China)",
    "Anti-China (Labor Migration China)",
    "Anti-Philippines (Social Economics)",
    "Anti-Western (US Military Philippines)",
]

def compute_correlation(data1, data2):
    """
    data1: {user: polarity}
    data2: {user: polarity}
    """
    common_keys = set(data1.keys()) & set(data2.keys())
    if len(common_keys) == 0:
        return 0
    print("Overlap", len(common_keys))
    data = []
    for key in common_keys:
        data.append([data1[key], data2[key]])
    data = np.array(data)
    corr = np.corrcoef(data[:, 0], data[:, 1])[0, 1]
    return corr


data = {}
for i in range(len(pro_map)):
    data[pro_map[i]] = {}
    data[anti_map[i]] = {}

for i, d in enumerate(dirs):
    with open(d + "/embedding.bin", "rb") as fin:
        emb = pickle.load(fin)
    with open(d + "/namelist.json", "r") as fin:
        namelist = json.load(fin)
    user_emb = emb[:len(namelist)]
    for k, user in enumerate(namelist):
        if emb[k, 0] >= emb[k, 1]:
            data[pro_map[i]][user] = emb[k, 0] / np.max(user_emb[:, 0])
        else:
            data[anti_map[i]][user] = emb[k, 1] / np.max(user_emb[:, 1])

corr_matrix = np.zeros(shape=(len(data), len(data)))
for i, (lname,l) in enumerate(data.items()):
    for j, (rname,r) in enumerate(data.items()):
        print(lname, rname)
        corr_matrix[i, j] = compute_correlation(l, r)


annotation = list(data.keys())

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=6)
matplotlib.rc('ytick', labelsize=6)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=annotation, yticklabels=annotation, cmap='coolwarm',annot_kws={"size": 7})
fig.subplots_adjust(bottom=0.2, left=0.2)
plt.title('Correlation Matrix Heatmap')
plt.yticks(rotation=70)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=340, ha="left", rotation_mode="anchor")
# plt.show()
plt.savefig("correlation.pdf")



