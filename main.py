"""
 python3 main.py --exp_name test --data_path datasets/combined_data.csv --device 1 --seed 0
"""
import os.path
import pickle
import random
import time

import torch
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from dataset import BeliefDataset
from model import ModelTrain
from evaluate import Evaluator

parser = argparse.ArgumentParser()

# General
parser.add_argument('--epochs', type=int, default=1000, help='epochs (iterations) for training')
parser.add_argument('--belief_warmup', type=int, default=200, help='epochs (iterations) for training')
parser.add_argument('--learning_rate', type=float, default=0.2, help='learning rate of model')
parser.add_argument('--device', type=str, default="cpu", help='cpu/gpu device')
parser.add_argument('--num_process', type=int, default=40, help='num_process for pandas parallel')

# Data
parser.add_argument('--exp_name', type=str, help='exp_name to use', required=True)
parser.add_argument('--dataset', type=str, help='dataset to use')
parser.add_argument('--data_path', type=str, default=None, help='specify the data path', required=True)
parser.add_argument('--pos_weight_lambda', type=float, default=1.0, help='Lambda for positive sample weight')
parser.add_argument('--save_freq', type=int, default=50, help='save_freq')

# For GAE/VGAE model
parser.add_argument('--polar_dim', type=int, default=2, help='polar_dim')
parser.add_argument('--belief_dim', type=int, default=7, help='belief_dim')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden_dim')
parser.add_argument('--temperature', type=float, default=0.1, help='smaller for shaper softmax for belief separation')
parser.add_argument('--belief_gamma', type=float, default=1.0, help='belief_gamma for weight of belief encoder loss')
parser.add_argument('--lr_cooldown', type=float, default=0.5, help='lr cooldown for belief encoder')
parser.add_argument('--seed', type=int, default=None)

args = parser.parse_args()
setattr(args, "output_path", Path(f"./belief_embedding_output_{args.exp_name}"))
args.output_path.mkdir(parents=True, exist_ok=True)

# Setting the device
if not torch.cuda.is_available():
    args.device = torch.device('cpu')
else:
    args.device = torch.device(int(args.device) if args.device.isdigit() else args.device)
print("Device: {}".format(args.device))

# Setting the random seeds
if args.seed is not None:
    print("set seed {}".format(args.seed))
    random.seed(a=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

# Prepare dataset (Use ApolloDataset for incas)
if not os.path.exists(args.output_path / "dataset.pkl"):
    with open(args.output_path / "dataset.pkl", "wb") as fout:
        dataset = BeliefDataset(data_path=args.data_path, args=args)
        dataset.build()
        pickle.dump(dataset, fout)
else:
    with open(args.output_path / "dataset.pkl", "rb") as fin:
        dataset = pickle.load(fin)
setattr(args, "num_user", dataset.num_user)
setattr(args, "num_assertion", dataset.num_assertion)
# dump label and namelist for evaluation
dataset.dump_data()

# Start Training
trainer = ModelTrain(dataset, args)
start_time = time.time()
trainer.train()
end_time = time.time()
running_time = end_time - start_time

exit()

# try:
#     feature = sp.diags([1.0], shape=(dataset.num_nodes, dataset.num_nodes))
#     setattr(args, "input_dim", dataset.num_nodes)
#     trainer = ControlVGAETrainer(adj_matrix, feature, args, dataset)
#     start_time = time.time()
#     trainer.train()
#     end_time = time.time()
#     running_time = end_time - start_time
#     trainer.save()
# except RuntimeError as e:
#     if "out of memory" in str(e).lower():
#         print("WARNING: ran out of vram, using cpu")
#         setattr(args, "device", "cpu")
#         feature = sp.diags([1.0], shape=(dataset.num_nodes, dataset.num_nodes))
#         setattr(args, "input_dim", dataset.num_nodes)
#         trainer = ControlVGAETrainer(adj_matrix, feature, args, dataset)
#         start_time = time.time()
#         trainer.train()
#         end_time = time.time()
#         running_time = end_time - start_time
#         trainer.save()
#     else:
#         raise e

# Start Evaluation (Apollo dataset)
print("Running Evaluation ...")
evaluator = Evaluator(use_b_matrix=args.use_b_matrix)
evaluator.init_from_value(trainer.result_embedding, dataset.user_label, dataset.asser_label,
                          dataset.name_list, dataset.tweetlist,
                          B_matrix=
                          None,
                          output_dir=args.output_path)
evaluator.plot(show=False, save=True)

evaluator.run_clustering()
evaluator.plot_clustering(show=False)
# evaluator.dump_text_result()
acc, macro_f1, avg_purity = evaluator.numerical_evaluate()

if not os.path.exists("records.csv"):
    with open("records.csv", "a") as fout:
        fout.write("dataset,semi_cnt_0,semi_cnt_1,num_assertion,seed,edge_guidance,axis_guidance,running_time,accuracy,macro_f1,average_purity\n")
with open("records.csv", "a") as fout:
    fout.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
        args.data_path,
        len(dataset.semi_indexs[0]),
        len(dataset.semi_indexs[1]),
        args.num_assertion,
        args.seed,
        args.edge_guidance,
        args.axis_guidance,
        running_time,
        acc,
        macro_f1,
        avg_purity
    ))

with open(args.output_path / "records.csv", "w") as fout:
    fout.write(
        "dataset,semi_cnt_0,semi_cnt_1,num_assertion,seed,edge_guidance,axis_guidance,running_time,accuracy,macro_f1,average_purity\n")
    fout.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
        args.data_path,
        len(dataset.semi_indexs[0]),
        len(dataset.semi_indexs[1]),
        args.num_assertion,
        args.seed,
        args.edge_guidance,
        args.axis_guidance,
        running_time,
        acc,
        macro_f1,
        avg_purity
    ))

with open("score.txt", "w") as fout:
    fout.write("{}".format(acc))

evaluator.dump_topk_json()
# evaluator.dump_topk_json_user()

# Dump top messages
m = dataset.original_tweetid2asserid
df = dataset.processed_data
lab = np.argmax(trainer.result_embedding, axis=1).tolist()
emb_val = np.max(trainer.result_embedding, axis=1).tolist()
df['cluster'] = df['tweet_id'].map(lambda x: lab[m[str(x)] + args.num_user])
df['emb_val'] = df['tweet_id'].map(lambda x: emb_val[m[str(x)] + args.num_user])
pd = pd.concat([
    df[df.cluster == i].sort_values(by=['emb_val', 'tweet_counts', 'user_counts', 'keyN'], ascending=False).drop_duplicates(subset='postTweet').iloc[:20] \
        for i in range(trainer.result_embedding.shape[1])
])
pd.to_pickle(args.output_path / 'top_messages.pkl')
pd.to_csv(args.output_path / 'top_messages.csv')
