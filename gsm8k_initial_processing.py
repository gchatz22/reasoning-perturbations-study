import json
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict

dataset = load_dataset("openai/gsm8k", "main")
split = "train"
dataset_split = dataset[split]

datapoints_by_reasoning_steps = defaultdict(list)
for i in tqdm(range(len(dataset_split))):
    datapoint = dataset_split[i]
    question = datapoint["question"]
    answer = datapoint["answer"]
    reasoning_steps = (
        len(answer.split("\n")) - 1
    )  # minus 1 due to the last new line being the answer
    datapoint["index"] = i
    datapoints_by_reasoning_steps[reasoning_steps].append(datapoint)

processed_datapoints_by_reasoning_steps = sorted(
    [
        {"steps": x, "datapoints": datapoints_by_reasoning_steps[x]}
        for x in datapoints_by_reasoning_steps
    ],
    key=lambda x: x["steps"],
)

path = "/Users/giannis/Desktop/datapoints_by_reasoning_steps.jsonl"
with open(path, "w") as file:
    for entry in processed_datapoints_by_reasoning_steps:
        file.write(json.dumps(entry) + "\n")
