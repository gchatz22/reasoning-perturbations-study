import json
import random

path = "/Users/giannis/Desktop/datapoints_by_reasoning_steps.jsonl"
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
