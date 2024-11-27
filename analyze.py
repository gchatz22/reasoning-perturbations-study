import re
import typer
from rich.console import Console
from rich.table import Table
from rich.align import Align

from constants import Perturbation, SEPARATOR

console = Console()


def main(
    model: str = typer.Option(help="Model to use for analysis"),
    perturbation: Perturbation = typer.Option(help="Perturbation to analyze"),
):
    if "/" in model:
        model = model.split("/")[1]

    path = "data/experiments/{}/cleaned-{}.txt".format(model, perturbation.value)
    data = None
    with open(path, "r") as file:
        data = file.read()

    datapoints = data.split(SEPARATOR)
    if datapoints[-1] == "\n":
        datapoints = datapoints[:-1]

    baseline_correct_count = 0
    experiment_correct_count = 0
    total_rows = 0

    table = Table(
        title="\n\n[bold]Results for {}, {} perturbation[/bold]".format(
            model, perturbation.value
        ),
        padding=(0, 2),
        show_footer=True,
    )
    table.add_column("ID", style="cyan bold")
    table.add_column("Correct Answer", justify="center", style="magenta bold")
    table.add_column("Baseline Answer", justify="center", style="magenta bold")
    table.add_column("Baseline Correctness", justify="center")
    table.add_column("Experiment Answer", justify="center", style="magenta bold")
    table.add_column("Experiment Correctness", justify="center")

    for datapoint in datapoints:
        idd = re.findall(r"ID:\s*(\d+)", datapoint)[0]
        extracted_correct_answer = re.findall(
            r">>>> Extracted Correct Answer:\s*(.*?)\n", datapoint
        )[0]
        extracted_baseline_response = re.findall(
            r">>>> Extracted Baseline Response:\s*(.*?)\n", datapoint
        )[0]
        extracted_experiment_response = re.findall(
            r">>>> Extracted Experiment Response:\s*(.*?)\n", datapoint
        )[0]

        if extracted_baseline_response == extracted_correct_answer:
            baseline_correct_count += 1
        if extracted_experiment_response == extracted_correct_answer:
            experiment_correct_count += 1
        total_rows += 1

        table.add_row(
            idd,
            extracted_correct_answer,
            extracted_baseline_response,
            ("✅" if extracted_baseline_response == extracted_correct_answer else "❌"),
            extracted_experiment_response,
            (
                "✅"
                if extracted_experiment_response == extracted_correct_answer
                else "❌"
            ),
        )

    baseline_percentage = (baseline_correct_count / total_rows) * 100
    experiment_percentage = (experiment_correct_count / total_rows) * 100

    table.columns[3].footer = f"{baseline_percentage:.2f}% Correct"
    table.columns[5].footer = f"{experiment_percentage:.2f}% Correct"

    console.print(Align.center(table))


if __name__ == "__main__":
    typer.run(main)
