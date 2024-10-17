import os
import time
import numpy as np

OUTPUT_DIR = None

def run_one_csv(csv_path, edge_guidance, axis_guidance, exp_name=None):
    if exp_name is None:
        exp_name = os.path.basename(csv_path).replace(".csv", "")

    for samping_ratio in (np.arange(0.05, 1 + 0.05, 0.05) if any([edge_guidance, axis_guidance]) else [0.0]):
        ratio = f"{samping_ratio:.2f},{samping_ratio:.2f}"
        if os.path.exists(f"{OUTPUT_DIR}/belief_embedding_output_{exp_name}_2_" + ratio.replace(",", "_")):
            continue
        best_score = -1
        for seed in range(10):
            os.system("rm score.txt")
            command = f"python3 main.py --data_path {csv_path} --exp_name {exp_name} " \
                      f"--hidden2_dim 2 --label_types supportive,opposing " \
                      f"--learning_rate 0.2  --label_sampling {ratio} --seed {seed} --device 1"
            if edge_guidance:
                command += " --edge_guidance"
            if axis_guidance:
                command += " --axis_guidance"
            print(command)
            if os.system(command) != 0:
                exit()
            with open("score.txt", "r") as fin:
                for line in fin:
                    score = float(line.strip())
                    if score > best_score:
                        print(f"Better! {score} {seed} {ratio}")
                        best_score = score
                        if os.path.exists(f"belief_embedding_output_{exp_name}_2_best"):
                            command = f"rm -r belief_embedding_output_{exp_name}_2_best"
                            if os.system(command) != 0:
                                exit()
                            time.sleep(3)
                        command = f"mv belief_embedding_output_{exp_name}_2 belief_embedding_output_{exp_name}_2_best"
                        if os.system(command) != 0:
                            exit()
                    else:
                        print(f"Worse! {score} {seed} {ratio} the best is: {best_score}")
        command = f"mv belief_embedding_output_{exp_name}_2_best {OUTPUT_DIR}/belief_embedding_output_{exp_name}_2_" + ratio.replace(",", "_")
        if os.system(command) != 0:
            exit()

import glob
csv_files = sorted(glob.glob("datasets/*.csv"))
for edge_guidance in [True, False]:
    for axis_guidance in [True, False]:
        OUTPUT_DIR = "SGVGAE"
        if edge_guidance:
            OUTPUT_DIR += "_Edge"
        if axis_guidance:
            OUTPUT_DIR += "_Axis"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for csv_file in csv_files:
            run_one_csv(csv_file, edge_guidance, axis_guidance)
