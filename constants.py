from enum import Enum


class Perturbation(Enum):
    IRRELEVANT = "irrelevant"
    PATHOLOGICAL = "pathological"
    RELEVANT = "relevant"
    COMBO = "combo"


TOTAL_SAMPLES = 50
SEPARATOR = "<SEPARATOR>"
