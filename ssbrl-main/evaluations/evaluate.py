import argparse
import os
from eval_metrics import calculate_metrics
import pandas as pd
import json

def evaluate(result_path, topic, model):
    csv = pd.read_csv(result_path)
    
    # Keep only the rows where is_gt = 1 and manual_label is not "neutral"
    csv = csv[(csv["is_gt"] == 1) & (csv["manual_label"] != "neutral")]
    csv = csv.drop_duplicates(subset="index_text", keep="first")
    
    if model in ["tweet_roberta_ft", "roberta_ft", "twhin_bert_ft"]:
        # Set ft_label to the opposite of manual_label for rows where gpt_label is "neutral"
        csv.loc[csv["ft_label"] == "neutral", "ft_label"] = csv.loc[csv["ft_label"] == "neutral", "manual_label"].apply(lambda x: "supportive" if x == "opposing" else "opposing")
    else:
        # Set gpt_label to the opposite of manual_label for rows where gpt_label is "neutral"
        csv.loc[csv["gpt_label"] == "neutral", "gpt_label"] = csv.loc[csv["gpt_label"] == "neutral", "manual_label"].apply(lambda x: "supportive" if x == "opposing" else "opposing")

    # Extract "manual_label" as ground truth and "gpt_label" as predictions
    if model in ["roberta_km", "tweet_roberta_km", "twhin_bert_km"]:
        predictions = csv["kmean_clsuter"]
    elif model in ["tweet_roberta_ft", "roberta_ft", "twhin_bert_ft"]:
        predictions = csv["ft_label"]
    else:
        predictions = csv["gpt_label"]
    ground_truth = csv["manual_label"] 
    
    # Print unique labels of ground_truth
    # unique_labels = ground_truth.unique()
    # unique_labels_pred = predictions.unique()
    # print("Unique labels of ground_truth:", unique_labels)
    # print("Unique labels of prediction:", unique_labels_pred)
    
    precision, recall, f1, acc, purity, oppo_pur, sur_pur = calculate_metrics(ground_truth, predictions)
    print(f"Evaluation result: precision: {precision}, recall: {recall}, f1: {f1}, accuracy: {acc}")
    
    # Create the folder if it doesn't exist
    if model in ["tweet_roberta_ft", "roberta_ft", "twhin_bert_ft"]:
        path = f"evaluation_result/finetune_result/{model}/"
    elif model in ["roberta_km", "tweet_roberta_km", "twhin_bert_km"]:
        path = f"evaluation_result/kmeans_res/{model}/"
    else:
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
    

def evaluate_model(model_name, result_csv, topic=""):
    print(f"Evaluating model: {model_name} on the topic {topic}")
    if model_name == "gpt-3":
        evaluate(result_csv, topic=topic, model="gpt-3")
    elif model_name == "gpt-4":
        evaluate(result_csv, topic=topic, model="gpt-4")
    elif model_name == "moe":
        evaluate(result_csv, topic=topic, model="moe")
    elif model_name == "roberta_km":
        evaluate(result_csv, topic=topic, model="roberta_km")
    elif model_name == "tweet_roberta_km":
        evaluate(result_csv, topic=topic, model="tweet_roberta_km")
    elif model_name == "twhin_bert_km":
        evaluate(result_csv, topic=topic, model="twhin_bert_km")
    elif model_name == "tweet_roberta_ft":
        evaluate(result_csv, topic=topic, model="tweet_roberta_ft")
    elif model_name == "roberta_ft":
        evaluate(result_csv, topic=topic, model="roberta_ft")
    elif model_name == "twhin_bert_ft":
        evaluate(result_csv, topic=topic, model="twhin_bert_ft")


def main():
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--model", required=True, type=str, help="Name of the model to evaluate")
    parser.add_argument("--csv", required=True, type=str, help="Result csv")
    parser.add_argument("--topic", required=True, type=str, help="Topic of the data")

    args = parser.parse_args()
    topic = args.topic
    evaluate_model(args.model, args.csv, topic)

if __name__ == "__main__":
    main()
