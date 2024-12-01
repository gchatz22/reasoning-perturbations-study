import os
import typer
from dotenv import load_dotenv
from rich.rule import Rule
from rich import print

from constants import Perturbation
from models import (
    OpenAIModel,
    AnthropicModel,
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    COHERE_MODELS,
    CohereModel,
    TogetherAIModel,
    TOGETHERAI_MODELS,
)
from main import (
    load_data,
    pre_processing_baseline,
    pre_processing_irrelavant,
    pre_processing_relevant,
    pre_processing_pathological,
    pre_processing_combo,
    validate_answer,
)

load_dotenv()


def main(
    model: str = typer.Option(help="Model to use for experiment"),
    perturbation: Perturbation = typer.Option(help="Perturbation to experiment with"),
    datapoint_index: int = typer.Option(help="Datapoint index to experiment with"),
):
    model_provider = None

    if model in OPENAI_MODELS:
        model_provider = OpenAIModel(
            api_key=os.getenv("OPENAI_API_KEY"), model_name=model
        )
    elif model in ANTHROPIC_MODELS:
        model_provider = AnthropicModel(
            api_key=os.getenv("ANTHROPIC_KEY"), model_name=model
        )
    elif model in COHERE_MODELS:
        model_provider = CohereModel(api_key=os.getenv("COHERE_KEY"), model_name=model)
    elif model in TOGETHERAI_MODELS:
        model_provider = TogetherAIModel(
            api_key=os.getenv("TOGETHER_AI_KEY"), model_name=model
        )

    if model_provider is None:
        raise Exception("Invalid input model.")

    dataset, samples_distribution = load_data()
    datapoint = [
        (x, batch["steps"])
        for batch in dataset
        for x in batch["datapoints"]
        if x["index"] == datapoint_index
    ]
    if not datapoint:
        raise Exception("Invalid index passed")

    datapoint, reasoning_steps = datapoint[0]

    correct_answer = datapoint["answer"]
    idd = datapoint["index"]

    print(
        "[bold red]>> Reasoning Steps: {}, ID: {}[/bold red]".format(
            reasoning_steps, idd
        )
    )

    baseline_prompt = pre_processing_baseline(datapoint)
    experiment_prompt = ""

    match perturbation:
        case Perturbation.IRRELEVANT:
            experiment_prompt = pre_processing_irrelavant(
                datapoint["question"], model_provider
            )
        case Perturbation.PATHOLOGICAL:
            experiment_prompt = pre_processing_pathological(
                datapoint["question"], model_provider
            )
        case Perturbation.RELEVANT:
            experiment_prompt = pre_processing_relevant(
                datapoint["question"], model_provider
            )
        case Perturbation.COMBO:
            experiment_prompt = pre_processing_combo(
                datapoint["question"], model_provider
            )

    print(
        "[green]>>> Question:[/green][white][not bold] {}[white not bold]\n".format(
            datapoint["question"]
        )
    )
    print(
        "[green]>>> Correct Answer:[/green][white not bold] {}[/white not bold]\n".format(
            datapoint["answer"]
        )
    )
    print(Rule(style="green"))
    baseline_response = model_provider.generate(prompt=baseline_prompt)
    print(
        "[green]>>> Baseline Answer:[/green][white not bold] {}[/white not bold]\n".format(
            baseline_response
        ),
    )
    print(Rule(style="green"))
    experiment_response = model_provider.generate(prompt=experiment_prompt)
    print(
        "[green]>>> Answer in {} experiment:[/green][white not bold] {}[/white not bold]\n".format(
            perturbation.value, experiment_response
        ),
    )
    validate_answer(correct_answer, baseline_response, experiment_response)


if __name__ == "__main__":
    typer.run(main)
