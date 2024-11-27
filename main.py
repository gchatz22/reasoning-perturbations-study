import re
import os
import json
import math
import typer
from enum import Enum
from rich import print
from rich.rule import Rule
from typing import Optional
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
    TogetherAIModel,
    TOGETHERAI_MODELS,
)


load_dotenv()
app = typer.Typer()


class Perturbation(Enum):
    IRRELEVANT = "irrelevant"
    PATHOLOGICAL = "pathological"
    RELEVANT = "relevant"


def load_data():
    parent_dir = "data/"
    path = parent_dir + "datapoints_by_reasoning_steps.jsonl"

    data = []
    with open(path, "r") as file:
        for line in file:
            data.append(json.loads(line.strip()))

    # NOTE: We are essentially taking the distribution and reducing the dataset granularity
    total_datapoints = sum([len(x["datapoints"]) for x in data])
    samples_distribution = {
        entry["steps"]: len(entry["datapoints"]) / total_datapoints for entry in data
    }

    return data, samples_distribution


def randomly_select_index(seen, dataset_size, exit_after=None):
    counter = 0
    while True:
        rand_index = randrange(dataset_size)
        if rand_index not in seen:
            return rand_index
        counter += 1
        if exit_after and counter > exit_after:
            return None


def pre_processing_baseline(datapoint):
    template = None
    with open("data/templates/baseline.txt", "r") as file:
        template = Template("".join(file.readlines()))

    question = datapoint["question"]
    prompt = template.substitute(question=question)
    return prompt


def log_experiment(model_name, perturbation, texts):
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    file_path = "data/experiments/{}/{}.txt".format(model_name, perturbation.value)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as file:
        for text in texts:
            file.write(text + "\n")
        file.write("<SEPARATOR>" + "\n")


def parse_seen_datapoints(model_name, perturbation):
    file_path = "data/experiments/{}/{}.txt".format(model_name, perturbation.value)
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = file.read()
            ids = re.findall(r"ID:\s*(\d+)", content)
            return set(ids)
    return set()


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
        seen.add(random_index)
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
        print("[red]Could not extract answers.[/red]")
        return extracted_correct_answer, "", ""
        # NOTE Using LLM to extract answer. Not consistent and spends $$, commenting out
        # since manual investigation of results will happen with way
        # print(
        #     "[red]Could not extract answers, extracting with an extractor model...[/red]"
        # )
        # extractor_provider = OpenAIModel(
        #     api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-4o"
        # )

        # template = None
        # with open("data/templates/extract_answers.txt", "r") as file:
        #     template = Template("".join(file.readlines()))

        # extractor_prompt = template.substitute(
        #     response1=extracted_baseline_response,
        #     response2=extracted_experiment_response,
        # )
        # extractor_response = extractor_provider.generate(prompt=extractor_prompt)
        # try:
        #     extracted_baseline_response, extracted_experiment_response = (
        #         extractor_response.split("@")
        #     )
        # except Exception:
        #     print(
        #         "[red]Could not extract answers. Raw model extractor response: {}[/red]".format(
        #             extractor_response
        #         )
        #     )
        #     return
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
    return (
        extracted_correct_answer,
        extracted_baseline_response,
        extracted_experiment_response,
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
    seen.add(random_index)
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
    restart_from_reasoning_steps: Optional[int] = typer.Option(
        None, help="Reasoning steps to restart from"
    ),
    restart_from_sample: Optional[int] = typer.Option(
        None, help="Sample to restart from"
    ),
):
    TOTAL_SAMPLES = 50
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
    samples = {k: math.ceil(TOTAL_SAMPLES * v) for k, v in samples_distribution.items()}
    print(
        "[bold red]>> Input number of samples was {} but resized to {} due to dataset distribution\n".format(
            TOTAL_SAMPLES, sum(samples.values())
        )
    )

    seen = parse_seen_datapoints(model_provider.model_name, perturbation)
    for reasoning_steps, num_samples in samples.items():
        if (
            restart_from_reasoning_steps
            and reasoning_steps < restart_from_reasoning_steps
        ):
            continue

        for i in range(num_samples):
            if (
                restart_from_reasoning_steps
                and reasoning_steps <= restart_from_reasoning_steps
                and restart_from_sample
                and i + 1 < restart_from_sample
            ):
                continue

            datapoints = [x for x in dataset if x["steps"] == reasoning_steps][0][
                "datapoints"
            ]
            random_index = randomly_select_index(seen, len(datapoints), exit_after=10)
            if random_index is None:
                # NOTE: This is a heuristic for reasoning steps distributions that only have a few samples
                # and are already logged as an experiment. Not ideal but works :).
                print(
                    "[bold red]>> Skipping {} reasoning steps sample because I couldn't find an unseen random index\n".format(
                        reasoning_steps
                    )
                )
                continue
            datapoint = datapoints[random_index]
            correct_answer = datapoint["answer"]
            idd = datapoint["index"]
            seen.add(idd)

            print(
                "[bold red]>> Reasoning Steps: {}, ID: {}, Sample {} out of {}[/bold red]".format(
                    reasoning_steps, idd, i + 1, num_samples
                )
            )

            baseline_prompt = pre_processing_baseline(datapoint)
            experiment_prompt = ""

            match perturbation:
                case Perturbation.IRRELEVANT:
                    experiment_prompt = pre_processing_irrelavant(
                        datapoint, model_provider
                    )
                case Perturbation.PATHOLOGICAL:
                    experiment_prompt = pre_processing_pathological(
                        datapoint, model_provider
                    )
                case Perturbation.RELEVANT:
                    experiment_prompt = pre_processing_relevant(
                        datapoint, model_provider
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
            (
                extracted_correct_answer,
                extracted_baseline_response,
                extracted_experiment_response,
            ) = validate_answer(correct_answer, baseline_response, experiment_response)
            print()
            print(Rule(style="red bold"))

            log_experiment(
                model_provider.model_name,
                perturbation,
                [
                    ">> Reasoning Steps: {}, ID: {}, Sample {} out of {}".format(
                        reasoning_steps, idd, i + 1, num_samples
                    ),
                    ">>> Question: {}".format(datapoint["question"]),
                    ">>> Correct Answer: {}".format(datapoint["answer"]),
                    ">>> Baseline Answer: {}".format(baseline_response),
                    ">>> Answer in {} experiment: {}".format(
                        perturbation.value, experiment_response
                    ),
                    ">>>> Extracted Correct Answer: {}".format(
                        extracted_correct_answer
                    ),
                    ">>>> Extracted Baseline Response: {}".format(
                        extracted_baseline_response
                    ),
                    ">>>> Extracted Experiment Response: {}".format(
                        extracted_experiment_response
                    ),
                ],
            )


if __name__ == "__main__":
    typer.run(main)
