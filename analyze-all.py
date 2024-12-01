import re
import os
from collections import defaultdict
import typer
from rich.console import Console
from rich.table import Table
from rich.align import Align

from constants import SEPARATOR

console = Console()
PERTURBATIONS = ["irrelevant", "relevant", "pathological", "combo"]


def analyze_by_reasoning_steps(data: str, model: str, perturbation: str):
    """
    Analyze the data and group results based on the number of reasoning steps.
    """
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

    table = Table(
        title=f"\n\n[bold]Results Breakdown by Reasoning Steps for {model}, {perturbation}[/bold]",
        padding=(0, 2),
    )
    table.add_column("Reasoning Steps", style="cyan bold", justify="center")
    table.add_column("Total Entries", style="magenta bold", justify="center")
    table.add_column("Baseline Accuracy (%)", justify="center")
    table.add_column("Experiment Accuracy (%)", justify="center")

    for steps, results in sorted(grouped_results.items()):
        total = results["total"]
        baseline_accuracy = (results["baseline_correct"] / total) * 100
        experiment_accuracy = (results["experiment_correct"] / total) * 100
        table.add_row(
            str(steps),
            str(total),
            f"{baseline_accuracy:.2f}",
            f"{experiment_accuracy:.2f}"
        )

    console.print(Align.center(table))


def main():
    base_path = "data/experiments"
    for model in os.listdir(base_path):
        model_path = os.path.join(base_path, model)
        if os.path.isdir(model_path):
            for perturbation in PERTURBATIONS:
                pattern = f"cleaned-{perturbation}.txt"
                for file in os.listdir(model_path):
                    if file == pattern:
                        file_path = os.path.join(model_path, file)
                        console.print(f"\nProcessing file: {file_path}\n")
                        with open(file_path, "r") as f:
                            data = f.read()
                        analyze_by_reasoning_steps(data, model, perturbation)


if __name__ == "__main__":
    typer.run(main)
