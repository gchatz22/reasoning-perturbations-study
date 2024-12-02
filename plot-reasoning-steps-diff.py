import re
import os
from collections import defaultdict
import typer
from rich.console import Console
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from constants import SEPARATOR


colors = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
    "#aec7e8",  # Light Blue
    "#ffbb78",  # Light Orange
    "#98df8a",  # Light Green
]


console = Console()
PERTURBATIONS = ["irrelevant", "relevant", "pathological", "combo"]
OUTPUT_DIR = "plots"


def analyze_by_reasoning_steps(data: str, model: str, perturbation: str):
    """Analyze the data and group results based on the number of reasoning steps."""
    datapoints = data.split(SEPARATOR)
    if datapoints[-1] == "\n":
        datapoints = datapoints[:-1]

    grouped_results = defaultdict(
        lambda: {"total": 0, "baseline_correct": 0, "experiment_correct": 0}
    )

    for datapoint in datapoints:
        reasoning_steps_match = re.search(r"Reasoning Steps:\s*(\d+)", datapoint)
        if reasoning_steps_match:
            correct_answer = re.findall(
                r">>>> Extracted Correct Answer:\s*(.*?)\n", datapoint
            )[0]
            baseline_response = re.findall(
                r">>>> Extracted Baseline Response:\s*(.*?)\n", datapoint
            )[0]
            experiment_response = re.findall(
                r">>>> Extracted Experiment Response:\s*(.*?)\n", datapoint
            )[0]

            reasoning_steps = reasoning_steps_match.group(1)
            if int(reasoning_steps) >= 7 and int(reasoning_steps) <= 11:
                reasoning_steps = ">7"

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


def create_legend_figure(labels, colors, markers, filename):
    """Create a separate figure for the legend with proper colors and markers."""
    fig_legend = plt.figure(figsize=(4, len(labels) * 0.5))
    for label, color, marker in zip(labels, colors, markers):
        plt.plot([], [], marker=marker, label=label, color=color, linestyle="None")
    plt.axis("off")
    plt.legend(loc="center", frameon=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}")
    plt.close()


def plot_percent_difference(perturbation: str, results):
    """Plot percent difference from baseline for the given perturbation."""
    percent_diff_labels = []
    percent_diff_markers = []
    # colormap = cm.get_cmap("Paired")  # Use a colormap (e.g., tab20)
    # colors = [colormap(i / len(results)) for i in range(len(results))]

    plt.figure(figsize=(12, 6))
    for idx, (
        model,
        (reasoning_steps, baseline_accuracies, experiment_accuracies),
    ) in enumerate(results.items()):
        percent_differences = [
            (experiment - baseline)
            for baseline, experiment in zip(baseline_accuracies, experiment_accuracies)
        ]
        marker = "s"
        plt.plot(
            reasoning_steps,
            percent_differences,
            marker=marker,
            label=model,
            color=colors[idx],
        )
        percent_diff_labels.append(model)
        percent_diff_markers.append(marker)

    plt.title(f"Percent Difference from Baseline - {perturbation.capitalize()}")
    plt.xlabel("Reasoning Steps")
    plt.ylabel("Percent Difference (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{perturbation}_percent_difference.png")
    plt.close()
    create_legend_figure(
        percent_diff_labels,
        colors,
        percent_diff_markers,
        f"{perturbation}_percent_difference_legend.png",
    )


def plot_aggregate_percent_difference(aggregated_results):
    """Plot aggregate percent difference for all perturbations."""
    percent_diff_labels = []
    percent_diff_markers = []
    colormap = cm.get_cmap("tab20")  # Use a colormap (e.g., tab20)
    colors = [
        colormap(i / len(aggregated_results)) for i in range(len(aggregated_results))
    ]

    plt.figure(figsize=(12, 6))
    for idx, (perturbation, data) in enumerate(aggregated_results.items()):
        reasoning_steps = sorted(data.keys())
        percent_differences = [
            (data[step]["experiment"] - data[step]["baseline"])
            for step in reasoning_steps
        ]
        marker = "s"
        plt.plot(
            reasoning_steps,
            percent_differences,
            marker=marker,
            label=perturbation.capitalize(),
            color=colors[idx],
        )
        percent_diff_labels.append(perturbation.capitalize())
        percent_diff_markers.append(marker)

    plt.title("Aggregate Percent Difference from Baseline vs. Reasoning Steps")
    plt.xlabel("Reasoning Steps")
    plt.ylabel("Percent Difference (%)")
    plt.xticks(range(2, 7))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/aggregate_percent_difference.png")
    plt.close()
    create_legend_figure(
        percent_diff_labels,
        colors,
        percent_diff_markers,
        "aggregate_percent_difference_legend.png",
    )


def main():
    base_path = "data/experiments"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    aggregated_results = {
        perturbation: defaultdict(lambda: {"baseline": 0, "experiment": 0, "count": 0})
        for perturbation in PERTURBATIONS
    }

    for perturbation in PERTURBATIONS:
        results = {}
        for model in sorted(
            [path.lower() for path in os.listdir(base_path)], reverse=True
        ):
            model_path = os.path.join(base_path, model)
            if os.path.isdir(model_path):
                pattern = f"cleaned-{perturbation}.txt"
                for file in os.listdir(model_path):
                    if file == pattern:
                        file_path = os.path.join(model_path, file)
                        console.print(
                            f"\n[white bold]Processing file: {file_path}[/white bold]"
                        )
                        with open(file_path, "r") as f:
                            data = f.read()
                        reasoning_steps, baseline_accuracies, experiment_accuracies = (
                            analyze_by_reasoning_steps(data, model, perturbation)
                        )
                        results[model] = (
                            reasoning_steps,
                            baseline_accuracies,
                            experiment_accuracies,
                        )

                        for i, step in enumerate(reasoning_steps):
                            aggregated_results[perturbation][step][
                                "baseline"
                            ] += baseline_accuracies[i]
                            aggregated_results[perturbation][step][
                                "experiment"
                            ] += experiment_accuracies[i]
                            aggregated_results[perturbation][step]["count"] += 1

        if results:
            # plot_results(perturbation, results)
            plot_percent_difference(perturbation, results)

    # normalize
    for perturbation, data in aggregated_results.items():
        for step, values in data.items():
            count = values["count"]
            if count > 0:
                values["baseline"] /= count
                values["experiment"] /= count

    # plot_aggregate_results(aggregated_results)
    plot_aggregate_percent_difference(aggregated_results)


if __name__ == "__main__":
    typer.run(main)
