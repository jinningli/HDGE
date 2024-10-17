acc_matrix = []
f1_matrix = []
purity_matrix = []
import numpy as np

with open("score.txt", "r", encoding="utf-8") as fin:
    for line in fin:
        if line.strip().split(" ")[0] == "Acc.":
            acc_matrix.append([])
            for s in line.strip().split(" ")[2:]:
                acc_matrix[-1].append(float(s.split("±")[0]))
        if line.strip().split(" ")[0] == "Macro":
            f1_matrix.append([])
            for s in line.strip().split(" ")[3:]:
                f1_matrix[-1].append(float(s.split("±")[0]))
        if line.strip().split(" ")[0] == "Purity":
            purity_matrix.append([])
            for s in line.strip().split(" ")[2:]:
                purity_matrix[-1].append(float(s.split("±")[0]))

print(np.array(acc_matrix).mean(axis=0))
print(np.array(f1_matrix).mean(axis=0))
print(np.array(purity_matrix).mean(axis=0))