from dataclasses import dataclass
from typing import Sequence
from kxor_code.algorithms.base_alg_step import (
    NoOpStep,
    ProblemRecord,
    StepConfig,
    StepStats,
    StepContext,
    BaseAlgorithmStep,
)
from kxor_code.algorithms.compute_kikuchi_step import ComputeKikuchiStep
from kxor_code.problem_set_generation.kxor_instance import KXORInstance


@dataclass
class AlgorithmPipeline:
    steps: Sequence[BaseAlgorithmStep]
    verbose: bool = False

    def run(self, problem: ProblemRecord):
        if self.verbose:
            print("--- Starting Algorithm Pipeline ---")
            print("Running pipeline on problem:", problem.problem_id)
            print("Pipeline steps:", [step.config.name for step in self.steps])

        for step in self.steps:
            for field in step.requires_fields:
                if not problem.has_field(field):
                    raise ValueError(
                        f"Problem is missing required fields for step {step.config.name}: {step.requires_fields}"
                    )
            step_stats = step.execute(problem)
