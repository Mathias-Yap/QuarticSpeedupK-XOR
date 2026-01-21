"""
Reusable scaffolding for composing algorithm steps that operate on batches of
problem instances.  Each step problems timings, intermediate quantities, and any
produced fields back onto the wrapped problem problems so downstream steps can
reason about provenance.
"""

from __future__ import annotations
import os
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Sequence

import numpy as np
from scipy import sparse

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
    
    def save(self, path: str) -> None:
        """Save the problem record to a file. Instance is saved as a separate KXORInstance file,
        and fields/metadata are saved in a companion JSON file that contains the instance path.
        The field kikuchi matrix is a sparse matrix and is saved separately in a .npz file. All files
        will be saved in a folder specified by the path argument.
        """
        folder_path = "/".join(path.split("/")[:-1])
        os.makedirs(folder_path, exist_ok=True) 
        self.instance.save(folder_path + "/kxor_instance.npz")
        np.savez_compressed(folder_path + "/kikuchi_matrix.npz", self.fields["kikuchi_matrix"])
        companion_path = folder_path + "/problem_records.json"
        self.fields.pop("kikuchi_matrix", None)  # Remove the sparse matrix before saving fields
        with open(companion_path, "w") as f:
            json.dump({
                "instance_path": path,
                "fields": self.fields,
                "step_history": [stat.as_dict() for stat in self.step_history]
            }, f)

    def save_compact(self, path: str, *, compress: bool = False) -> None:
        """Save the problem record to a single .npz file for faster reloads."""
        payload: Dict[str, Any] = {
            "schema_version": np.array(1, dtype=np.int32),
            "problem_id": np.array(self.problem_id),
            "instance_n": np.array(self.instance.n, dtype=np.int64),
            "instance_k": np.array(self.instance.k, dtype=np.int64),
            "instance_m": np.array(self.instance.m, dtype=np.int64),
            "instance_scopes": np.asarray(self.instance.scopes),
            "instance_b": np.asarray(self.instance.b),
            "instance_is_planted": np.array(self.instance.is_planted),
            "instance_rho": np.array(-1.0 if self.instance.rho is None else self.instance.rho),
            "instance_has_rho": np.array(self.instance.rho is not None),
            "instance_z": np.asarray(self.instance.z) if self.instance.z is not None else np.array([]),
            "instance_has_z": np.array(self.instance.z is not None),
        }

        array_fields: List[str] = []
        sparse_fields: List[str] = []
        json_fields: Dict[str, Any] = {}

        for key, value in self.fields.items():
            if sparse.issparse(value):
                csr = value.tocsr()
                sparse_fields.append(key)
                payload[f"field_sparse__{key}__data"] = csr.data
                payload[f"field_sparse__{key}__indices"] = csr.indices
                payload[f"field_sparse__{key}__indptr"] = csr.indptr
                payload[f"field_sparse__{key}__shape"] = np.array(csr.shape, dtype=np.int64)
                payload[f"field_sparse__{key}__format"] = np.array(value.getformat())
            elif isinstance(value, np.ndarray):
                array_fields.append(key)
                payload[f"field_array__{key}"] = value
            else:
                try:
                    json.dumps(value, default=self._json_default)
                except TypeError as exc:
                    raise TypeError(
                        f"Field '{key}' is not JSON-serializable; store numpy arrays directly "
                        "or extend save_compact serialization."
                    ) from exc
                json_fields[key] = value

        payload["fields_meta"] = np.array(
            json.dumps({"array_fields": array_fields, "sparse_fields": sparse_fields})
        )
        payload["fields_json"] = np.array(json.dumps(json_fields, default=self._json_default))

        try:
            step_history_json = json.dumps(
                [stat.as_dict() for stat in self.step_history],
                default=self._json_default,
            )
        except TypeError as exc:
            raise TypeError(
                "Step history contains non-JSON-serializable values; consider normalizing "
                "StepStats.additional_data before saving."
            ) from exc
        payload["step_history_json"] = np.array(step_history_json)

        saver = np.savez_compressed if compress else np.savez
        saver(path, **payload)

    @staticmethod
    def load_compact(path: str) -> ProblemRecord:
        """Load a problem record saved via save_compact."""
        data = np.load(path, allow_pickle=False)
        schema_version = int(data["schema_version"]) if "schema_version" in data else 0
        if schema_version != 1:
            raise ValueError(f"Unsupported problem record schema version: {schema_version}")

        has_rho = bool(data["instance_has_rho"])
        rho = None if not has_rho else float(data["instance_rho"])
        has_z = bool(data["instance_has_z"])
        z = data["instance_z"]
        if not has_z:
            z = None

        instance = KXORInstance.create(
            scopes=data["instance_scopes"],
            b=data["instance_b"],
            is_planted=bool(data["instance_is_planted"]),
            rho=rho,
            z=z,
        )

        fields_meta_raw = ProblemRecord._coerce_str(data["fields_meta"].item())
        fields_meta = json.loads(fields_meta_raw)
        array_fields = fields_meta.get("array_fields", [])
        sparse_fields = fields_meta.get("sparse_fields", [])

        fields_json_raw = ProblemRecord._coerce_str(data["fields_json"].item())
        fields: Dict[str, Any] = json.loads(fields_json_raw) if fields_json_raw else {}

        for key in array_fields:
            fields[key] = data[f"field_array__{key}"]

        for key in sparse_fields:
            data_arr = data[f"field_sparse__{key}__data"]
            indices = data[f"field_sparse__{key}__indices"]
            indptr = data[f"field_sparse__{key}__indptr"]
            shape_raw = data[f"field_sparse__{key}__shape"]
            shape = tuple(int(x) for x in shape_raw)
            fmt = ProblemRecord._coerce_str(data[f"field_sparse__{key}__format"].item())
            csr = sparse.csr_matrix((data_arr, indices, indptr), shape=shape)
            fields[key] = ProblemRecord._convert_sparse_format(csr, fmt)

        step_history_raw = ProblemRecord._coerce_str(data["step_history_json"].item())
        step_history_data = json.loads(step_history_raw) if step_history_raw else []
        step_history = [
            StepStats(
                step_name=stat["step"],
                version=stat["version"],
                problem_id=stat["problem_id"],
                started_at=stat["started_at"],
                completed_at=stat["completed_at"],
                failed=stat["failed"],
                additional_data=stat.get("additional_data", {}),
            )
            for stat in step_history_data
        ]

        problem_id = ProblemRecord._coerce_str(data["problem_id"].item())
        return ProblemRecord(
            problem_id=problem_id,
            instance=instance,
            fields=fields,
            step_history=step_history,
        )

    @staticmethod
    def _convert_sparse_format(matrix: sparse.csr_matrix, fmt: str) -> sparse.spmatrix:
        if fmt == "dok":
            return matrix.todok()
        if fmt == "csc":
            return matrix.tocsc()
        if fmt == "coo":
            return matrix.tocoo()
        return matrix

    @staticmethod
    def _coerce_str(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    @staticmethod
    def load(path: str) -> ProblemRecord:
        """Load a problem record from a file."""
        companion_path = path + ".json"
        with open(companion_path, "r") as f:
            data = json.load(f)
        instance = KXORInstance.load(data["instance_path"])
        problem = ProblemRecord(
            problem_id="loaded_problem",
            instance=instance,
            fields=data.get("fields", {}),
            step_history=[
                StepStats(
                    step_name=stat["step"],
                    version=stat["version"],
                    problem_id=stat["problem_id"],
                    started_at=stat["started_at"],
                    completed_at=stat["completed_at"],
                    failed=stat["failed"],
                    additional_data=stat.get("additional_data", {})
                ) for stat in data.get("step_history", [])
            ]
        )
        return problem



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

        # Keep step logging self-contained by default (StreamHandler + any explicitly attached
        # handlers). This avoids surprising duplication via root handlers in notebook contexts.
        self._logger.propagate = False

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
