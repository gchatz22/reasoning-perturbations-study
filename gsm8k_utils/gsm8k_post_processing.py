import math
import json
import random
import pandas as pd

random.seed(10)

parent_dir = (
    "/Users/giannis/Desktop/[6.861] Quantitative Methods for NLP/final_project/data/"
)
path = parent_dir + "datapoints_by_reasoning_steps.jsonl"
data = []
with open(path, "r") as file:
    for line in file:
        data.append(json.loads(line.strip()))

# NOTE: We are essentially taking the distribution and reducing the dataset granularity
total_datapoints = sum([len(x["datapoints"]) for x in data])
samples = {
    entry["steps"]: len(entry["datapoints"]) / total_datapoints for entry in data
}
total_number_of_problems = 50

for entry in data:
    reasoning_steps = entry["steps"]
    datapoints = entry["datapoints"]
    num_samples = math.ceil(total_number_of_problems * samples[reasoning_steps]) 

    print("Reasoning steps:", reasoning_steps)
    random_indices = random.sample(range(len(datapoints)), num_samples)
    random_samples = [datapoints[i] for i in random_indices]

    df = pd.DataFrame(random_samples)
    path = parent_dir + "/reasoning_steps/reasoning-steps-{}.csv".format(
        reasoning_steps
    )
    df.to_csv(path)

    # for i, sample in enumerate(random_samples, 1):
    #     print(f"Sample {i}:")
    #     print("Index:", sample["index"])
    #     print()
    #     print("Question:", sample["question"])
    #     print()
    #     print("Answer:", sample["answer"])
    #     input()

    # print(
    #     "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    # )
