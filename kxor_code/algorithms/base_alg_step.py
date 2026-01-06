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
    notes: Dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        if self.completed_at is None:
            self.completed_at = time.perf_counter()

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at is None:
            return None
        return self.completed_at - self.started_at

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
            "notes": dict(self.notes),
        }


@dataclass
class StepContext:
    """Context shared across all problems in a batch while a step runs."""

    rng: random.Random
    problem_id: str
    logger: logging.Logger
    scratch: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemRecord:
    """Wraps a single problem instance with metadata and provenance."""

    problem_id: str
    instance: Any
    fields: Dict[str, Any] = field(default_factory=dict)
    intermediates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    step_history: List[StepStats] = field(default_factory=list)
    """StepStats instance containing data for each step that has run on this problem."""

    
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

    def add_intermediate(self, step_name: str, key: str, value: Any) -> None:
        self.intermediates.setdefault(step_name, {})[key] = value

    # def problem_step(
    #     self,
    #     step_name: str,
    #     *,
    #     duration_seconds: float,
    #     success: bool,
    #     produced_fields: Optional[Sequence[str]] = None,
    #     error: Optional[str] = None,
    # ) -> None:
    #     entry = {
    #         "step": step_name,
    #         "timestamp": time.time(),
    #         "duration_seconds": duration_seconds,
    #         "success": success,
    #     }
    #     if produced_fields:
    #         entry["produced_fields"] = list(produced_fields)
    #     if error is not None:
    #         entry["error"] = error
    #     self.step_history.append(entry)

    def add_metadata(self, stats: StepStats) -> None:
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
            "Starting step '%s' (version %s) on problem %s (fields=%d)",
            self.name,
            self.config.version,
            problem.problem_id,
        )

        stats.started_at = time.perf_counter()
        produced_fields: Optional[Dict[str, Any]] = None
        error_msg: Optional[str] = None
        try:
            produced_fields = self._run(problem, context) or {}
            for key, value in produced_fields.items():
                problem.add_field(key, value)
        except Exception as exc:  
            stats.failed = True
            error_msg = str(exc)
            problem.add_intermediate(self.name, "error", error_msg)
            self._logger.exception(
                "Step '%s' failed for problem '%s'", self.name, problem.problem_id
            )
            if self.raise_on_error:
                stats.finish()
                raise
        finally:
            duration = time.perf_counter() - stats.started_at
            stats.completed_at = time.perf_counter()
            # problem.step_history.append(stats)

        stats.finish()
        problem.add_metadata(stats)
        self._logger.info(
            "Finished step '%s' in %.3fs (processed=%d, failed=%d, problem=%s)",
            self.name,
            stats.duration_seconds or -1.0,
            stats.failed,
            problem.problem_id,
        )
        return stats

    def add_intermediate(self, problem: ProblemRecord, key: str, value: Any) -> None:
        """Helper for subclasses to problem intermediate data on the problem."""
        problem.add_intermediate(self.name, key, value)

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
        self, problem: ProblemRecord, context: StepContext
    ) -> Optional[Dict[str, Any]]:
        """
        Execute the step for a single problem.

        Return a mapping of produced fields that should be attached to the
        problem problem.  Subclasses may also call `add_intermediate` for richer
        trace data.
        """


class NoOpStep(BaseAlgorithmStep):
    """Simple illustrative step that only problems the runtime."""

    def _run(self, problem: ProblemRecord, context: StepContext) -> Optional[Dict[str, Any]]:
        self.add_intermediate(problem, "noop", f"touched by {self.name}")
        context.logger.debug("NoOpStep touched problem %s", problem.problem_id)
        return {}
