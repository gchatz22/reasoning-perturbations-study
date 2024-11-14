import json
import random

import os

path = "/Users/giannis/Desktop/[6.861] Quantitative Methods for NLP/final_project/datapoints_by_reasoning_steps.jsonl"
datapoints = []
with open(path, "r") as file:
    for line in file:
        datapoints.append(json.loads(line.strip()))

for entry in datapoints:
    print("Reasoning steps:", entry["steps"])
    print("Amount:", len(entry["datapoints"]))
    print()
# num_samples = len(dataset[split])
# sample_count = 10
# random_indices = random.sample(range(num_samples), sample_count)
# random_samples = [dataset[split][i] for i in random_indices]

# for i, sample in enumerate(random_samples, 1):
#     print(f"Sample {i}:")
#     print("Question:", sample["question"])
#     print()
#     print("Answer:", sample["answer"])
#     print(
#         "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
#     )
