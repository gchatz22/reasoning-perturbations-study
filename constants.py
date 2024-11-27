from enum import Enum


class Perturbation(Enum):
    IRRELEVANT = "irrelevant"
    PATHOLOGICAL = "pathological"
    RELEVANT = "relevant"


TOTAL_SAMPLES = 50
SEPARATOR = "<SEPARATOR>"