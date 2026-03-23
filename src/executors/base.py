from dataclasses import dataclass


@dataclass(slots=True)
class RunContext:
    experiment_name: str
    output_dir: str
