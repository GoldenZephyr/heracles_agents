from dataclasses import dataclass


@dataclass
class ModelInfo:
    model: str
    temperature: float
    seed: int
