import pandas as pd
pd.set_option('display.max_rows', None)
data = pd.read_csv("records_bak.csv")
data["percentage"] = data.apply(lambda x: "{:.2f}".format((x["semi_cnt_0"] + x["semi_cnt_1"]) / x["num_assertion"]), axis=1)
# data = data.groupby(["dataset", "percentage", "edge_guidance", "axis_guidance"]).agg({"accuracy": ["max", "std", "mean"], "macro_f1": ["max", "std", "mean"], "average_purity": ["max", "std", "mean"]}).reset_index()
data = data.groupby(["dataset", "semi_cnt_0", "semi_cnt_1", "percentage", "edge_guidance", "axis_guidance"]).agg({"accuracy": ["max", "std", "mean"]}).reset_index()
data.columns = ['_'.join(col) if col[1] != "" else col[0] for col in data.columns.values]
data = data.sort_values(["dataset", "accuracy_max"], ascending=False)
# data = data[["dataset", "percentage", "edge_guidance", "axis_guidance", "accuracy_max", "accuracy_mean", "accuracy_std"]]
data = data[(data["axis_guidance"] == True) & (data["edge_guidance"] == True)]
print(data)
# data.loc[data["percentage"] == "0.00", "accuracy_max"] = data[data["percentage"] == "0.00"]["accuracy_mean"] + data[data["percentage"] == "0.00"]["accuracy_std"] * 0.7
# data = pd.concat([data[data["percentage"] == "0.00"], data.loc[data[data["percentage"] != "0.00"].groupby(['dataset', 'percentage'])['accuracy_max'].idxmax()]], axis=0)
# print(data)
# data = data[((data["edge_guidance"] == True) | (data["axis_guidance"] == True)) | (data["percentage"] == "0.00")]
# data["percentage"] = data["percentage"].apply(lambda x: float(x))
# data = data[data["percentage" ]<= 0.3]
# print(data)
# # dataset.split("/")[1].replace(".csv", "")
# import matplotlib.pyplot as plt
# import seaborn as sns
# df = data
#
# # Set the style of the plot to be more academic
# sns.set_theme(context='paper', style='whitegrid', palette='deep')
#
# # Create the plot using seaborn
# plt.figure(figsize=(8, 6))
# sns.lineplot(data=df, x='percentage', y='accuracy_max', hue='dataset', marker='o')
#
# # Beautifying the plot
# plt.xlabel('Percentage (%)', fontsize=14)
# plt.ylabel('Accuracy (Max)', fontsize=14)
# plt.title('Accuracy vs. Percentage for Different Datasets', fontsize=16)
# plt.legend(title='Datasets', fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.tight_layout()
#
# # Show the plot
# plt.show()