import re
import os
import json
import typer
from enum import Enum
from rich import print
from rich.rule import Rule
from string import Template
from random import randrange
from dotenv import load_dotenv

from models import (
    OpenAIModel,
    AnthropicModel,
    OPENAI_MODELS,
    ANTHROPIC_MODELS,
    COHERE_MODELS,
    CohereModel,
)


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
            seen.add(rand_index)
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
    number_of_irrelevant_context = 0
    prompt = template.substitute(question=question, irrelevant_text=irrelevant_text)
    tokenized_prompt = model_provider.tokenize(prompt)
    while len(tokenized_prompt) < token_limit:
        random_index = randomly_select_index(seen, len(files))
        random_file_name = files[random_index]
        with open(directory + random_file_name, "r") as file:
            text = "".join(file.readlines())
            irrelevant_text += text
            number_of_irrelevant_context += len(model_provider.tokenize(text))

        prompt = template.substitute(question=question, irrelevant_text=irrelevant_text)
        tokenized_prompt = model_provider.tokenize(prompt)

    if len(tokenized_prompt) > token_limit:
        diff = len(tokenized_prompt) - token_limit
        extra_token_ids = tokenized_prompt[-diff:]
        extra = len(model_provider.detokenize(extra_token_ids))
        prompt = prompt[:-extra]
        number_of_irrelevant_context -= len(extra_token_ids)

    print(
        "[bold red]>> Number of additional tokens:[/bold red][white][not bold] {}[/white][/not bold]\n".format(
            number_of_irrelevant_context
        )
    )

    return prompt


def validate_answer(correct_answer, baseline_response, experiment_response):
    # try to extract with regex
    pattern = r"####\s*([^\s]+)"
    extracted_correct_answer = re.findall(pattern, correct_answer)[0]
    extracted_baseline_response = re.findall(pattern, baseline_response)
    extracted_experiment_response = re.findall(pattern, experiment_response)

    if len(extracted_baseline_response) != 1 or len(extracted_experiment_response) != 1:
        print(
            "[red]Could not extract answers, extracting with an extractor model...[/red]"
        )
        extractor_provider = OpenAIModel(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o"
        )

        template = None
        with open("data/templates/extract_answers.txt", "r") as file:
            template = Template("".join(file.readlines()))

        extractor_prompt = template.substitute(
            response1=extracted_baseline_response,
            response2=extracted_experiment_response,
        )
        extractor_response = extractor_provider.generate(prompt=extractor_prompt)
        try:
            extracted_baseline_response, extracted_experiment_response = (
                extractor_response.split("@")
            )
        except Exception:
            print(
                "[red]Could not extract answers. Raw model extractor response: {}[/red]".format(
                    extractor_response
                )
            )
            return
    else:
        extracted_baseline_response = extracted_baseline_response[0].strip(" ")
        extracted_experiment_response = extracted_experiment_response[0].strip(" ")

    print(
        "[green bold]Correct response: {}[/green bold]".format(extracted_correct_answer)
    )
    print(
        "[green bold]Baseline response: {} {}[/green bold]".format(
            extracted_baseline_response,
            "✅" if extracted_baseline_response == extracted_correct_answer else "❌",
        )
    )
    print(
        "[green bold]Experiment response: {} {}[/green bold]".format(
            extracted_experiment_response,
            "✅" if extracted_experiment_response == extracted_correct_answer else "❌",
        )
    )


def pre_processing_pathological(datapoint, model_provider):
    template = None
    with open("data/templates/pathological.txt", "r") as file:
        template = Template("".join(file.readlines()))

    question = datapoint["question"]
    pathologies = []
    with open("data/pathological/data.txt", "r") as file:
        pathologies = [line.strip("\n") for line in file.readlines()]

    seen = set()
    random_index = randomly_select_index(seen, len(pathologies))
    pathology = pathologies[random_index]

    print(
        "[bold red]>> Pathology:[/bold red][white][not bold] {}[/white][/not bold]\n".format(
            pathology
        )
    )

    prompt = template.substitute(question=question, pathology=pathology)
    return prompt


def pre_processing_relevant(datapoint, model_provider):
    template = None
    with open("data/templates/relevant.txt", "r") as file:
        template = Template("".join(file.readlines()))

    question = datapoint["question"]
    metaprompt = template.substitute(question=question)

    augmented_prompt = model_provider.generate(prompt=metaprompt)
    print(
        "[bold red]>> Augmented prompt:[/bold red][white][not bold] {}[/white][/not bold]\n".format(
            augmented_prompt
        )
    )

    with open("data/templates/baseline.txt", "r") as file:
        template = Template("".join(file.readlines()))

    augmented_prompt = template.substitute(question=augmented_prompt)
    return augmented_prompt


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
    elif model in COHERE_MODELS:
        model_provider = CohereModel(api_key=os.getenv("COHERE_KEY"), model_name=model)

    if model_provider is None:
        raise Exception("Invalid input model.")

    dataset = load_data()
    match perturbation:
        case Perturbation.IRRELEVANT:
            seen = set()
            for i in range(random_samples):
                random_index = randomly_select_index(seen, len(dataset))
                print(
                    "[bold red]>> Sample {} out of {}, ID {}[/bold red]".format(
                        i + 1, random_samples, random_index
                    )
                )
                datapoint = dataset[random_index]
                baseline_prompt = pre_processing_baseline(datapoint)
                experiment_prompt = pre_processing_irrelavant(datapoint, model_provider)
                print(
                    "[green]>>> Question:[/green][white][not bold] {}[white not bold]\n".format(
                        datapoint["question"]
                    ),
                )
                print(
                    "[green]>>> Correct Answer:[/green][white not bold] {}[/white not bold]\n".format(
                        datapoint["answer"]
                    )
                )
                print(Rule(style="green"))
                baseline_response = model_provider.generate(prompt=baseline_prompt)
                print(
                    "[green]>>> Answer w/o irrelevant context:[/green][white not bold] {}[/white not bold]\n".format(
                        baseline_response
                    ),
                )
                print(Rule(style="green"))
                experiment_response = model_provider.generate(prompt=experiment_prompt)
                print(
                    "[green]>>> Answer with irrelevant context:[/green][white not bold] {}[/white not bold]\n".format(
                        experiment_response
                    ),
                )
                print(Rule(style="red bold"))
        case Perturbation.PATHOLOGICAL:
            seen = set()
            for i in range(random_samples):
                random_index = randomly_select_index(seen, len(dataset))
                print(
                    "[bold red]>> Sample {} out of {}, ID {}[/bold red]".format(
                        i + 1, random_samples, random_index
                    )
                )
                datapoint = dataset[random_index]
                correct_answer = datapoint["answer"]
                baseline_prompt = pre_processing_baseline(datapoint)
                experiment_prompt = pre_processing_pathological(
                    datapoint, model_provider
                )
                print(Rule(style="green"))
                print(
                    "[green]>>> Question:[/green][white not bold] {}[/white not bold]\n".format(
                        datapoint["question"]
                    ),
                )
                print()
                print(
                    "[green]>>> Correct Answer:[/green][white not bold] {}[/white not bold]\n".format(
                        correct_answer
                    )
                )
                print(Rule(style="green"))
                baseline_response = model_provider.generate(prompt=baseline_prompt)
                print(
                    "[green]>>> Answer w/o pathology:[/green][white not bold] {}[/white not bold]\n".format(
                        baseline_response
                    ),
                )
                print(Rule(style="green"))
                experiment_response = model_provider.generate(prompt=experiment_prompt)
                print(
                    "[green]>>> Answer with pathology:[/green][white not bold] {}[/white not bold]\n".format(
                        experiment_response
                    ),
                )
                validate_answer(correct_answer, baseline_response, experiment_response)
                print(Rule(style="red bold"))
        case Perturbation.RELEVANT:
            seen = set()
            for i in range(random_samples):
                random_index = randomly_select_index(seen, len(dataset))
                print(
                    "[bold red]>> Sample {} out of {}, ID {}[/bold red]".format(
                        i + 1, random_samples, random_index
                    )
                )
                datapoint = dataset[random_index]
                baseline_prompt = pre_processing_baseline(datapoint)
                experiment_prompt = pre_processing_relevant(datapoint, model_provider)
                print(Rule(style="green"))
                print(
                    "[green]>>> Question:[/green][white not bold] {}[/white not bold]\n".format(
                        datapoint["question"]
                    ),
                )
                print()
                print(
                    "[green]>>> Correct Answer:[/green][white not bold] {}[/white not bold]\n".format(
                        datapoint["answer"]
                    )
                )
                print(Rule(style="green"))
                baseline_response = model_provider.generate(prompt=baseline_prompt)
                print(
                    "[green]>>> Answer with baseline:[/green][white not bold] {}[/white not bold]\n".format(
                        baseline_response
                    ),
                )
                print(Rule(style="green"))
                experiment_response = model_provider.generate(prompt=experiment_prompt)
                print(
                    "[green]>>> Answer with augmented prompt:[/green][white not bold] {}[/white not bold]\n".format(
                        experiment_response
                    ),
                )
                print(Rule(style="red bold"))


if __name__ == "__main__":
    typer.run(main)
