"""
Reusable scaffolding for composing algorithm steps that operate on batches of
problem instances.  Each step problems timings, intermediate quantities, and any
produced fields back onto the wrapped problem problems so downstream steps can
reason about provenance.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence

from kxor_code.problem_set_generation.kxor_instance import KXORInstance


@dataclass(frozen=True)
class StepConfig:
    """Immutable configuration for a base algorithm step."""

    name: str
    version: str = "0.1"
    seed: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def updated(self, **overrides: Any) -> "StepConfig":
        """Return a copy of this config with overrides applied."""
        return replace(self, **overrides)


@dataclass
class StepStats:
    """Aggregated runtime information for a step over a batch."""

    step_name: str
    version: str
    problem_id: str
    started_at: float = field(default_factory=time.perf_counter)
    completed_at: Optional[float] = None
    failed: bool = False
    additional_data: dict = field(default_factory=dict)

    def finish(self) -> None:
        if self.completed_at is None:
            self.completed_at = time.perf_counter()

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return self.completed_at - self.started_at
    
    def add_data(self, key: str, value: Any) -> None:
        """Add additional data to the step stats."""
        self.additional_data[key] = value

    
    
    def as_dict(self) -> Dict[str, Any]:
        """Convenience helper for serialising stats."""
        return {
            "step": self.step_name,
            "version": self.version,
            "problem_id": self.problem_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "failed": self.failed,
            "additional_data": self.additional_data,
        }


@dataclass
class StepContext:
    """Context of a problem run step."""

    rng: random.Random
    """Random number generator for the step if applicable."""
    problem_id: str
    """ID of the problem being processed."""
    logger: logging.Logger
    """Logger for the step."""

@dataclass
class ProblemRecord:
    """Wraps a single problem instance with metadata and provenance."""

    problem_id: str
    instance: KXORInstance
    fields: Dict[str, Any] = field(default_factory=dict)
    step_history: List[StepStats] = field(default_factory=list)
    """StepStats instance containing basic data for each step that has run on this problem."""

    def has_field(self, key: str) -> bool:
        return key in self.fields

    def add_field(self, key: str, value: Any) -> None:
        self.fields[key] = value

    def get_field(self, key: str, default: Any = None) -> Any:
        return self.fields.get(key, default)

    def expect_fields(self, keys: Iterable[str]) -> None:
        missing = [key for key in keys if key not in self.fields]
        if missing:
            raise KeyError(
                f"Problem '{self.problem_id}' is missing required field(s): {missing}"
            )
    
    def add_metadata(self, stats: StepStats) -> None:
        """Add step stats metadata to the problem record."""
        self.step_history.append(stats)



class BaseAlgorithmStep(ABC):
    """
    Base class for algorithm steps that operate on a ProblemRecord.

    Sub-classes should implement `_run` and optionally extend hooks to emit
    intermediates.  The base class wires together execution, validation, timing,
    and provenance probleming.
    """

    requires_fields: Sequence[str] = ()
    produces_fields: Sequence[str] = ()
    raise_on_error: bool = False

    def __init__(self, config: Optional[StepConfig] = None, *, logger: Optional[logging.Logger] = None) -> None:
        if config is None:
            config = StepConfig(name=self.__class__.__name__)
        self.config = config
        self._logger = logger or logging.getLogger(self.config.name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    @property
    def name(self) -> str:
        return self.config.name

    def execute(self, problem: ProblemRecord) -> StepStats:
        problem.expect_fields(self.requires_fields)
        stats = StepStats(step_name=self.name, version=self.config.version, problem_id=problem.problem_id)
        context = self._build_context(problem)
        self._logger.info(
            "Starting step '%s' (version %s) on problem %s (fields=%s)",
            self.name,
            self.config.version,
            problem.problem_id,
            problem.fields
        )

        stats.started_at = time.perf_counter()
        produced_fields: Optional[Dict[str, Any]] = None
        error_msg: Optional[str] = None
        try:
            produced_fields= self._run(problem, context, stats) or {} 
            for key, value in produced_fields.items(): #type: ignore
                problem.add_field(key, value)

        except Exception as exc:  
            stats.failed = True
            error_msg = str(exc)
            stats.add_data("error", error_msg)
            self._logger.exception(
                "Step '%s' failed for problem '%s'", self.name, problem.problem_id
            )
            if self.raise_on_error:
                stats.finish()
                raise
        finally:
            duration = time.perf_counter() - stats.started_at
            stats.completed_at = time.perf_counter()
            stats.duration_seconds
            # problem.step_history.append(stats)

        stats.finish()
        problem.add_metadata(stats)
        self._logger.info(
            "Finished step '%s' in %.3fs (success: %s, problem=%s)",
            self.name,
            stats.duration_seconds or -1.0,
            not stats.failed,
            problem.problem_id,
        )
        return stats

    def ensure_problem_fields(self, problem: ProblemRecord, keys: Iterable[str]) -> None:
        problem.expect_fields(keys)

    def _build_context(self, problem: ProblemRecord) -> StepContext:
        rng = (
            random.Random(self.config.seed)
            if self.config.seed is not None
            else random.Random()
        )
        rng.seed((self.config.seed or uuid.uuid4().int) ^ hash(problem.problem_id)) #type: ignore
        return StepContext(rng=rng, problem_id=problem.problem_id, logger=self._logger)

    @abstractmethod
    def _run(
        self, problem: ProblemRecord, context: StepContext, stats: StepStats
    ) -> Optional[Dict[str, Any]]: 
        """ 
        Execute the step for a single problem. Return a mapping of produced fields that should be attached to the
        problem problem. For collecting data on step execution, use the provided StepStats instance's `add_data` method.
        
        return {"produced_field": value}
        """


class NoOpStep(BaseAlgorithmStep):
    """Simple illustrative step that only problems the runtime."""

    def _run(self, problem: ProblemRecord, context: StepContext, stats: StepStats) -> Optional[Dict[str, Any]]:
        stats.add_data("message", "No operation performed.")
        context.logger.debug("NoOpStep touched problem %s", problem.problem_id)
        return {}
