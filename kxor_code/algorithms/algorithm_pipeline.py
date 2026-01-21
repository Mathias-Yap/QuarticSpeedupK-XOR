from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, Sequence
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
    logger: Optional[logging.Logger] = None
    log_file: Optional[str] = None
    log_level: int = logging.INFO

    _HANDLER_TAG = "_kxor_pipeline_file_handler"

    def _attach_file_handler(self, logger: logging.Logger, log_file: str) -> None:
        """Attach a tagged FileHandler if not already present."""
        for handler in logger.handlers:
            if getattr(handler, self._HANDLER_TAG, False):
                return

        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        handler.setLevel(self.log_level)
        handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        setattr(handler, self._HANDLER_TAG, True)
        logger.addHandler(handler)

    def run(self, problem: ProblemRecord):
        logger = self.logger or logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.log_level)

        if self.log_file:
            self._attach_file_handler(logger, self.log_file)
            for step in self.steps:
                step_logger = getattr(step, "_logger", None)
                if isinstance(step_logger, logging.Logger):
                    step_logger.setLevel(self.log_level)
                    self._attach_file_handler(step_logger, self.log_file)
        if self.verbose:
            logger.info("--- Starting Algorithm Pipeline ---")
            logger.info("Running pipeline on problem: %s", problem.problem_id)
            logger.info("Pipeline steps: %s", [step.config.name for step in self.steps])

        for step in self.steps:
            for field in step.requires_fields:
                if not problem.has_field(field):
                    raise ValueError(
                        f"Problem is missing required fields for step {step.config.name}: {step.requires_fields}"
                    )
            step_stats = step.execute(problem)
