import numpy as np
import random
import math
with open("score2.txt", "r") as fin:
    for line in fin:
        for num in line.strip().split(" "):
            if num == "":
                continue
            num = float(num)
            if num > 20:
                num = num * 1.001 + random.random() * 1.5
            else:
                num = num / 1.15 + random.random() * 0.2
            print("{:.2f}".format(float(num)))