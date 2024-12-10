import re
import os
import typer
from rich import print
from rich.rule import Rule
from rich.table import Table
from rich.align import Align
from rich.console import Console

from constants import SEPARATOR

console = Console()


def calculate_na(path):
    data = None
    with open(path, "r") as file:
        data = file.read()

    datapoints = data.split(SEPARATOR)
    if datapoints[-1] == "\n":
        datapoints = datapoints[:-1]

    count, total_count = 0, 0
    for datapoint in datapoints:
        if datapoint == "" or datapoint == "\n":
            break
        extracted_experiment_response = re.findall(
            r">>>> Extracted Experiment Response:\s*(.*?)\n", datapoint
        )[0]
        if "n/a" in extracted_experiment_response:
            count += 1
        total_count += 1

    return count, total_count


def main():
    experiments_dir = "data/experiments"
    table = Table(
        title="\n\n[bold]n/a results[/bold]",
        padding=(0, 2),
        show_footer=True,
    )
    table.add_column("Model", justify="center", style="cyan bold")
    table.add_column("Irrelevant", justify="center", style="magenta bold")
    table.add_column("Pathological", justify="center", style="magenta bold")
    table.add_column("Relevant", justify="center", style="magenta bold")
    table.add_column("Combo", justify="center", style="magenta bold")

    for model in os.listdir(experiments_dir):
        model_experiments_dir = "{}/{}/".format(experiments_dir, model)
        percentages = {}
        for experiment in os.listdir(model_experiments_dir):
            if "cleaned" in experiment:
                # print(
                #     "Model: [white bold]{}[/white bold], Experiment: [white bold]{}[/white bold]".format(
                #         model, experiment.split("-")[1].split(".")[0]
                #     )
                # )
                experiment_path = "{}/{}/{}".format(experiments_dir, model, experiment)
                count, total_count = calculate_na(experiment_path)
                # print(Rule(style="red"))
                # print()
                percentage = f"{(count / total_count) * 100:.2f}%"
                percentages[experiment.split("-")[1].split(".")[0]] = str(percentage)
        table.add_row(
            model,
            percentages.get("irrelevant", ""),
            percentages["pathological"],
            percentages["relevant"],
            percentages["combo"],
        )
    console.print(Align.center(table))


if __name__ == "__main__":
    typer.run(main)
