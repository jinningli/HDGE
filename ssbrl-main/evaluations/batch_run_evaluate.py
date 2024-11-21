import os
import argparse
import pandas as pd
import json
from eval_metrics import calculate_metrics
from tqdm import tqdm
import time

avg_precision = 0
avg_recall = 0
avg_f1 = 0
avg_acc = 0
avg_purity = 0
avg_oppo_pur = 0
avg_sur_pur = 0
total_time = 0

def evaluate(result_path, topic, model):
    global avg_precision, avg_recall, avg_f1, avg_acc, avg_purity, avg_oppo_pur, avg_sur_pur
    csv = pd.read_csv(result_path)
    
    # Keep only the rows where is_gt = 1 and manual_label is not "neutral"
    csv = csv[(csv["is_gt"] == 1) & (csv["manual_label"] != "neutral")]

    csv = csv.drop_duplicates(subset="index_text", keep="first")
    
    # Set gpt_label to the opposite of manual_label for rows where gpt_label is "neutral"
    csv.loc[csv["gpt_label"] == "neutral", "gpt_label"] = csv.loc[csv["gpt_label"] == "neutral", "manual_label"].apply(lambda x: "supportive" if x == "opposing" else "opposing")
    
    # Keep only rows with gpt_label = supportive or opposing
    csv = csv[csv["gpt_label"].isin(["supportive", "opposing"])]

    # Extract "manual_label" as ground truth and "gpt_label" as predictions
    ground_truth = csv["manual_label"] 
    predictions = csv["gpt_label"]
        
    precision, recall, f1, acc, purity, oppo_pur, sur_pur = calculate_metrics(ground_truth, predictions)
    print(f"Evaluation result: precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {acc}")
    avg_precision += precision
    avg_recall += recall
    avg_f1 += f1
    avg_acc += acc
    avg_purity += purity
    avg_oppo_pur += oppo_pur
    avg_sur_pur += sur_pur

    # Create the folder if it doesn't exist
    path = f"evaluation_result/batch_run_result/{model}/"
    os.makedirs(path, exist_ok=True)

    # Save the result into a JSON file
    result = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "opposing_purity": oppo_pur,
        "supportive_purity": sur_pur,
        "average_purity": purity
    }
    
    file_path = os.path.join(path, f"{model}-{topic}.json")
    with open(file_path, "a") as file:
        json.dump(result, file, indent=4)
        file.write(",\n")
        print(f"Saved the evaluation result to {file_path}")
    

# Command for running labeling:
# python3 sentiment.py --topic insurgent_threats --data '/Users/ruipenghan/projects/research/ssbrl/data/paper_data_merged/filtered_data_2_2_2_20000_tree_vis_insurgent-threats_labeled_merged.csv' --instruction '/Users/ruipenghan/projects/research/ssbrl/baselines/prompts/insurgent_threats.txt'  --dest_folder labeled_data --key OPENAI_API_KEY --result_mapping insurgent_threats.json --model gpt-4-1106-preview

def batch_run_eval_gpt(args):
    global avg_precision, avg_recall, avg_f1, avg_acc, avg_purity, avg_oppo_pur, avg_sur_pur, total_time
    topic = args.topic
    raw_data = args.data
    instruction = args.instruction
    key = args.key
    result_mapping = args.result_mapping
    model = args.model
    command = f"python3 /Users/ruipenghan/projects/research/ssbrl/baselines/gpt-4/sentiment.py" \
              f" --topic {topic} --data {raw_data}" \
              f" --instruction {instruction} --dest_folder labeled_data --key {key}" \
              f" --result_mapping {result_mapping}  --model {model}"
    print(command)

    result_path = f"labeled_data/{topic}/"
    result_csv_file = None
    # Run and evaluate 10 times
    for i in tqdm(range(10)):
        start_time = time.time()  # Start the timer
        if os.system(command) != 0:
            print(f"Encountered issues at run {i}, exiting")
            exit(1)
        end_time = time.time()  # Start the timer
        total_time += (end_time - start_time)
        if result_csv_file is None:
            for file in os.listdir(result_path):
                if file.endswith(".csv"):
                    result_csv_file = os.path.join(result_path, file)
                    break
        # print(result_csv_file)
        # Evaluation:
        evaluate(result_csv_file, topic, model)
        # break

    # Save the result into a JSON file
    average_result = {
        "average_precision": avg_precision / 10.0,
        "average_recall": avg_recall / 10.0,
        "average_f1": avg_f1 / 10.0,
        "average_accuracy": avg_acc / 10.0,
        "average_opposing_purity": avg_oppo_pur / 10.0,
        "average_supportive_purity": avg_sur_pur / 10.0,
        "overall_average_purity": avg_purity / 10.0,
        "average_run_time": total_time / 10.0
    }
    path = f"evaluation_result/batch_run_result/{model}/"
    file_path = os.path.join(path, f"{model}-{topic}.json")
    with open(file_path, "a") as file:
        json.dump(average_result, file, indent=4)
        
    print("Finished all 10 evaluations")


# Command for running moe:
# python3 mixtral.py --topic us_military_philippines --data '/Users/ruipenghan/projects/research/ssbrl/data/paper_data_merged/filtered_data_10_5_5_20000_tree_vis_us_military-philippine_labeled_merged.csv' --instruction '/Users/ruipenghan/projects/research/ssbrl/baselines/prompts/us_military-philippine.txt'  --result_mapping us_military_philippine.json --worker 1 --dest_folder labeled_data
    
def batch_run_eval_moe(args):
    global avg_precision, avg_recall, avg_f1, avg_acc, avg_purity, avg_oppo_pur, avg_sur_pur, total_time
    topic = args.topic
    raw_data = args.data
    instruction = args.instruction
    result_mapping = args.result_mapping
    model = args.model
    command = f"python3 /Users/ruipenghan/projects/research/ssbrl/baselines/moe/mixtral.py" \
              f" --topic {topic} --data {raw_data}" \
              f" --instruction {instruction} --dest_folder labeled_data" \
              f" --result_mapping {result_mapping}  --worker 1"
    print(command)

    result_path = f"labeled_data/{topic}/"
    result_csv_file = None
    # Run and evaluate 10 times
    for i in tqdm(range(10)):
        start_time = time.time()  # Start the timer
        if os.system(command) != 0:
            print(f"Encountered issues at run {i}, exiting")
            exit(1)
        end_time = time.time()  # Start the timer
        total_time += (end_time - start_time)
        if result_csv_file is None:
            for file in os.listdir(result_path):
                if file.endswith(".csv"):
                    result_csv_file = os.path.join(result_path, file)
                    break
        # print(result_csv_file)
        # Evaluation:
        evaluate(result_csv_file, topic, model)
        # break

    # Save the result into a JSON file
    average_result = {
        "average_precision": avg_precision / 10.0,
        "average_recall": avg_recall / 10.0,
        "average_f1": avg_f1 / 10.0,
        "average_accuracy": avg_acc / 10.0,
        "average_opposing_purity": avg_oppo_pur / 10.0,
        "average_supportive_purity": avg_sur_pur / 10.0,
        "overall_average_purity": avg_purity / 10.0,
        "average_run_time": total_time / 10.0
    }
    path = f"evaluation_result/batch_run_result/{model}/"
    file_path = os.path.join(path, f"{model}-{topic}.json")
    with open(file_path, "a") as file:
        json.dump(average_result, file, indent=4)
        
    print("Finished all 10 evaluations")


def main():
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--model", required=True, type=str, help="Name of the model to evaluate; gpt-3.5-turbo-1106 or gpt-4-1106-preview, or moe")
    parser.add_argument('--topic', type=str, required=True, help='The current topic')
    
    parser.add_argument('--data', type=str, required=True, default='', help='the path to the raw csv file containing the tweets')

    parser.add_argument('--key', type=str, default="OPENAI_API_KEY", help='RUIJIE_KEY, WALLY_KEY, or OPENAI_API_KEY')

    parser.add_argument('--result_mapping', type=str, required=True, default="result_mapping.json", help='Result is saved to this file')

    parser.add_argument('--instruction', type=str, required=True, help='instruction for the prompt to GPT model. Default is prompts/edca_prompt.txt')


    args = parser.parse_args()
    if args.model == 'gpt-3.5-turbo-1106':
        batch_run_eval_gpt(args)
    elif args.model == 'gpt-4-1106-preview':
        batch_run_eval_gpt(args)
    elif args.model == 'moe':
        batch_run_eval_moe(args)
    else:
        print("INVALID MODEL CHOICE. EXITING")
        exit(1)
    
    return


if __name__ == "__main__":
    main()

