import os
import json
import typer
from enum import Enum
from string import Template
from random import randrange
from dotenv import load_dotenv

from models import OpenAIModel, AnthropicModel, OPENAI_MODELS, ANTHROPIC_MODELS


load_dotenv()
app = typer.Typer()


class Perturbation(Enum):
    IRRELEVANT = "irrelevant"
    PATHOLOGICAL = "pathological"
    RELEVANT = "relevant"


def load_data():
    with open("data/gsm8k_test_split.jsonl", "r") as file:
        return [json.loads(line.strip()) for line in file]


def randomly_select_index(seen, dataset_size):
    while True:
        rand_index = randrange(dataset_size)
        if rand_index not in seen:
            return rand_index


def pre_processing_baseline(datapoint):
    template = None
    with open("data/templates/baseline.txt", "r") as file:
        template = Template("".join(file.readlines()))

    question = datapoint["question"]
    prompt = template.substitute(question=question)
    return prompt


def pre_processing_irrelavant(datapoint, model_provider):
    template = None
    with open("data/templates/irrelevant.txt", "r") as file:
        template = Template("".join(file.readlines()))

    question = datapoint["question"]
    # NOTE: utilize 90% of context window to prevent potential overflow
    token_limit = round(model_provider.max_token_limit() * 0.9)

    directory = "data/irrelevant/"
    files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]

    seen = set()
    irrelevant_text = ""
    prompt = template.substitute(question=question, irrelevant_text=irrelevant_text)
    tokenized_prompt = model_provider.tokenize(prompt)
    while len(tokenized_prompt) < token_limit:
        random_index = randomly_select_index(seen, len(files))
        random_file_name = files[random_index]
        with open(directory + random_file_name, "r") as file:
            text = "".join(file.readlines())
            irrelevant_text += text

        prompt = template.substitute(question=question, irrelevant_text=irrelevant_text)
        tokenized_prompt = model_provider.tokenize(prompt)

    if len(tokenized_prompt) > token_limit:
        diff = len(tokenized_prompt) - token_limit
        extra_token_ids = tokenized_prompt[-diff:]
        extra = len(model_provider.detokenize(extra_token_ids))
        prompt = prompt[:-extra]

    return prompt


def main(
    model: str = typer.Option(help="Model to use for experiment"),
    perturbation: Perturbation = typer.Option(help="Perturbation to experiment with"),
    random_samples: int = typer.Option(
        help="Number of random samples to experiment with"
    ),
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

    if model_provider is None:
        raise Exception("Invalid input model.")

    dataset = load_data()
    match perturbation:
        case Perturbation.IRRELEVANT:
            seen = set()
            for i in range(random_samples):
                print(">> Sample {}".format(i + 1))
                random_index = randomly_select_index(seen, len(dataset))
                datapoint = dataset[random_index]
                baseline_prompt = pre_processing_baseline(datapoint)
                experiment_prompt = pre_processing_irrelavant(datapoint, model_provider)
                print(">>> Question:", datapoint["question"], "\n")
                print(">>> Correct Answer:", datapoint["answer"], "\n")
                baseline_response = model_provider.generate(prompt=baseline_prompt)
                print(">>> Answer w/o irrelevant context:", baseline_response, "\n")
                experiment_response = model_provider.generate(prompt=experiment_prompt)
                print(">>> Answer with irrelevant context:", experiment_response, "\n")
                print()
        case Perturbation.PATHOLOGICAL:
            print("yooo")
        case Perturbation.RELEVANT:
            print("temp")


if __name__ == "__main__":
    typer.run(main)