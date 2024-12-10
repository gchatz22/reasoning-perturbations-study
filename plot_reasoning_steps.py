from main import load_data
import matplotlib.pyplot as plt

data, _ = load_data()
data = [(x["steps"], len(x["datapoints"])) for x in data]

x_values, y_values = zip(*data)
plt.bar(x_values, y_values, color="skyblue", width=0.6)

plt.xlabel("Reasoning Steps")
plt.ylabel("Count")
plt.title("Count of data points per answer reasoning steps")
plt.xticks(x_values)
plt.grid(axis="y", linestyle="--", alpha=0.7)
for x, y in zip(x_values, y_values):
    plt.text(x, y + 1, str(y), ha="center", fontsize=8)
plt.savefig("plots/reasoning_steps_count.png")
