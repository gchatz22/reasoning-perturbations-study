import re
import os
from collections import defaultdict
import typer
from rich.console import Console
import matplotlib.pyplot as plt
from constants import SEPARATOR

console = Console()
PERTURBATIONS = ["irrelevant", "relevant", "pathological", "combo"]
OUTPUT_DIR = "plots"

def analyze_by_reasoning_steps(data: str, model: str, perturbation: str):
    """Analyze the data and group results based on the number of reasoning steps."""
    datapoints = data.split(SEPARATOR)
    if datapoints[-1] == "\n":
        datapoints = datapoints[:-1]
        
    grouped_results = defaultdict(lambda: {
        "total": 0,
        "baseline_correct": 0,
        "experiment_correct": 0
    })

    for datapoint in datapoints:
        reasoning_steps_match = re.search(r"Reasoning Steps:\s*(\d+)", datapoint)
        if reasoning_steps_match:
            reasoning_steps = int(reasoning_steps_match.group(1))
            if reasoning_steps < 2 or reasoning_steps > 6:  # Limit to steps between 2 and 6
                continue

            correct_answer = re.findall(
                r">>>> Extracted Correct Answer:\s*(.*?)\n", datapoint
            )[0]
            baseline_response = re.findall(
                r">>>> Extracted Baseline Response:\s*(.*?)\n", datapoint
            )[0]
            experiment_response = re.findall(
                r">>>> Extracted Experiment Response:\s*(.*?)\n", datapoint
            )[0]

            grouped_results[reasoning_steps]["total"] += 1
            if baseline_response == correct_answer:
                grouped_results[reasoning_steps]["baseline_correct"] += 1
            if experiment_response == correct_answer:
                grouped_results[reasoning_steps]["experiment_correct"] += 1

    reasoning_steps_list = sorted(grouped_results.keys())
    baseline_accuracies = []
    experiment_accuracies = []

    for steps in reasoning_steps_list:
        results = grouped_results[steps]
        total = results["total"]
        baseline_accuracies.append((results["baseline_correct"] / total) * 100)
        experiment_accuracies.append((results["experiment_correct"] / total) * 100)

    return reasoning_steps_list, baseline_accuracies, experiment_accuracies

def create_legend_figure(labels, markers, filename):
    """Create a separate figure for the legend."""
    fig_legend = plt.figure(figsize=(4, len(labels) * 0.5))
    for label, marker in zip(labels, markers):
        plt.plot([], [], marker=marker, label=label)
    plt.axis('off')
    plt.legend(loc='center', frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()

def plot_results(perturbation: str, results):
    """Plot results for the given perturbation."""
    # Baseline Accuracy
    baseline_labels = []
    baseline_markers = []
    plt.figure(figsize=(12, 6))
    for model, (reasoning_steps, baseline_accuracies, _) in results.items():
        marker = 'o'
        plt.plot(reasoning_steps, baseline_accuracies, marker=marker, label=model)
        baseline_labels.append(model)
        baseline_markers.append(marker)
    plt.title(f"Baseline Accuracy - {perturbation.capitalize()}")
    plt.xlabel("Reasoning Steps")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(2, 7))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{perturbation}_baseline.png")
    plt.close()
    create_legend_figure(baseline_labels, baseline_markers, f"{perturbation}_baseline_legend.png")

    # Experiment Accuracy
    experiment_labels = []
    experiment_markers = []
    plt.figure(figsize=(12, 6))
    for model, (reasoning_steps, _, experiment_accuracies) in results.items():
        marker = 'x'
        plt.plot(reasoning_steps, experiment_accuracies, marker=marker, label=model)
        experiment_labels.append(model)
        experiment_markers.append(marker)
    plt.title(f"Experiment Accuracy - {perturbation.capitalize()}")
    plt.xlabel("Reasoning Steps")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(2, 7))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{perturbation}_experiment.png")
    plt.close()
    create_legend_figure(experiment_labels, experiment_markers, f"{perturbation}_experiment_legend.png")

def plot_aggregate_results(aggregated_results):
    """Plot aggregate results for all perturbations."""
    # Baseline Accuracy
    baseline_labels = []
    baseline_markers = []
    plt.figure(figsize=(12, 6))
    for perturbation, data in aggregated_results.items():
        reasoning_steps = sorted(data.keys())
        baseline_accuracies = [data[step]["baseline"] for step in reasoning_steps]
        marker = 'o'
        plt.plot(reasoning_steps, baseline_accuracies, marker=marker, label=perturbation.capitalize())
        baseline_labels.append(perturbation.capitalize())
        baseline_markers.append(marker)
    plt.title("Aggregate Baseline Accuracy vs. Reasoning Steps")
    plt.xlabel("Reasoning Steps")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(2, 7))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aggregate_baseline.png")
    plt.close()
    create_legend_figure(baseline_labels, baseline_markers, "aggregate_baseline_legend.png")

    # Experiment Accuracy
    experiment_labels = []
    experiment_markers = []
    plt.figure(figsize=(12, 6))
    for perturbation, data in aggregated_results.items():
        reasoning_steps = sorted(data.keys())
        experiment_accuracies = [data[step]["experiment"] for step in reasoning_steps]
        marker = 'x'
        plt.plot(reasoning_steps, experiment_accuracies, marker=marker, label=perturbation.capitalize())
        experiment_labels.append(perturbation.capitalize())
        experiment_markers.append(marker)
    plt.title("Aggregate Experiment Accuracy vs. Reasoning Steps")
    plt.xlabel("Reasoning Steps")
    plt.ylabel("Accuracy (%)")
    plt.xticks(range(2, 7))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aggregate_experiment.png")
    plt.close()
    create_legend_figure(experiment_labels, experiment_markers, "aggregate_experiment_legend.png")

def main():
    base_path = "data/experiments"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aggregated_results = {perturbation: defaultdict(lambda: {"baseline": 0, "experiment": 0, "count": 0}) for perturbation in PERTURBATIONS}

    for perturbation in PERTURBATIONS:
        results = {}
        for model in os.listdir(base_path):
            model_path = os.path.join(base_path, model)
            if os.path.isdir(model_path):
                pattern = f"cleaned-{perturbation}.txt"
                for file in os.listdir(model_path):
                    if file == pattern:
                        file_path = os.path.join(model_path, file)
                        console.print(f"\nProcessing file: {file_path}\n")
                        with open(file_path, "r") as f:
                            data = f.read()
                        reasoning_steps, baseline_accuracies, experiment_accuracies = analyze_by_reasoning_steps(
                            data, model, perturbation
                        )
                        results[model] = (reasoning_steps, baseline_accuracies, experiment_accuracies)
                        
                        # Aggregate data
                        for i, step in enumerate(reasoning_steps):
                            aggregated_results[perturbation][step]["baseline"] += baseline_accuracies[i]
                            aggregated_results[perturbation][step]["experiment"] += experiment_accuracies[i]
                            aggregated_results[perturbation][step]["count"] += 1
        
        if results:
            plot_results(perturbation, results)

    # Normalize aggregate data
    for perturbation, data in aggregated_results.items():
        for step, values in data.items():
            count = values["count"]
            if count > 0:
                values["baseline"] /= count
                values["experiment"] /= count

    plot_aggregate_results(aggregated_results)

if __name__ == "__main__":
    typer.run(main)
