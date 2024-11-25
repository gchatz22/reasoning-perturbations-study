import json
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

dataset = load_dataset("openai/gsm8k", "main")
split = "test"
dataset_split = dataset[split]

parent_dir = (
    "/Users/giannis/Desktop/[6.861] Quantitative Methods for NLP/final_project/data/"
)
path = parent_dir + "gsm8k_test_split.jsonl"

with open(path, "w") as file:
    for i in tqdm(range(len(dataset_split))):
        datapoint = dataset_split[i]
        file.write(json.dumps(datapoint) + "\n")
