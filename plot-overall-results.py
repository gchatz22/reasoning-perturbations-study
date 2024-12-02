import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    "Model": [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "command-r-plus-08-2024",
        "gemma-2-9b-it",
        "gemma-2-27b-it",
        "gpt-4o",
        "o1-preview",
        "Llama-3.2-3B-Instruct-Turbo",
        "Meta-Llama-3.1-8B-Instruct-Turbo",
        "Meta-Llama-3.1-405B-Instruct-Turbo",
        "Mistral-7B-Instruct-v0.3",
        "Mixtral-8x22B-Instruct-v0.1"
    ],
    "Relevant": [
        -5.36, -3.57, -5.36, -10.72, -5.36, -3.57, 0.00, -3.57, -19.64, -10.72, 0.00, -5.36, -17.86
    ],
    "Irrelevant": [
        0.00, 0.00, 0.00, -67.86, -42.86, -83.93, -62.50, -3.57, -53.57, -75.00, -51.73, -39.29, -78.58 
    ],
    "Pathological": [
        -1.79, 0.00, -1.79, -10.71, -16.07, -8.96, 0.00, -1.79, -8.93, -14.28, -5.35, -19.64, -21.43
    ],
    "Combo": [
        -1.79, 0.00, -14.28, -41.07, -10.71, -12.50, -1.78, -1.78, -17.86, -12.50, -3.57, -28.57, -21.43
    ],
}

df = pd.DataFrame(data)

x = np.arange(len(df["Model"]))  
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.bar(x - 1.5 * width, df["Relevant"], width, label="Relevant")
bar2 = ax.bar(x - 0.5 * width, df["Irrelevant"], width, label="Irrelevant")
bar3 = ax.bar(x + 0.5 * width, df["Pathological"], width, label="Pathological")
bar4 = ax.bar(x + 1.5 * width, df["Combo"], width, label="Combo")

ax.set_xlabel("Models", fontsize=12)
ax.set_ylabel("Performance Drop (%)", fontsize=12)
ax.set_title("Performance Drop by Model and Prompt Type", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df["Model"], rotation=45, ha="right", fontsize=10)
ax.legend()

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()