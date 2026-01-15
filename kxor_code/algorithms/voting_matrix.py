from __future__ import annotations

import itertools
import time
from math import comb
from typing import Any, Mapping

import numpy as np


def form_voting_matrix_common_remainder(
	v_top: np.ndarray | Mapping[tuple[int, ...], complex],
	*,
	n: int,
	ell: int,
	subset_list: list[tuple[int, ...]] | None = None,
	subset_index: dict[tuple[int, ...], int] | None = None,
	logger: Any | None = None,
) -> np.ndarray:
	r"""Compute the n x n voting matrix V(v_top) using the "common remainder R" formula.

	For i < j:
		V[i,j] = 0.5 * sum_{R \subset [n]\{i,j}, |R|=ell-1} conj(v_{R∪{i}}) * v_{R∪{j}}
	V is Hermitian with zeros on diagonal.

	Parameters
	----------
	v_top:
		Either an array of length C(n, ell) ordered by `subset_list` (default: lexicographic
		`itertools.combinations(range(n), ell)`), or a dict-like mapping from subset tuples
		to complex values.
	n, ell:
		Problem parameters.
	subset_list, subset_index:
		Optional precomputed subset ordering/indexing.
	logger:
		Optional logger with `.info(...)`.
	"""
	n = int(n)
	ell = int(ell)
	if n <= 0:
		raise ValueError("n must be positive")
	if ell <= 0 or ell > n:
		raise ValueError("ell must satisfy 1 <= ell <= n")

	t0 = time.perf_counter()

	if isinstance(v_top, np.ndarray):
		v_top_arr = np.asarray(v_top, dtype=complex).reshape(-1)
		expected_len = comb(n, ell)
		if v_top_arr.size < expected_len:
			raise ValueError(
				f"v_top numpy array length {v_top_arr.size} is smaller than comb({n},{ell})={expected_len}."
			)
		if subset_index is None:
			if subset_list is None:
				subset_list = list(itertools.combinations(range(n), ell))
			subset_index = {subset: idx for idx, subset in enumerate(subset_list)}
		v_top_map: Mapping[tuple[int, ...], complex] | None = None
	else:
		v_top_arr = None
		v_top_map = v_top
		if subset_list is None:
			subset_list = list(v_top_map.keys())

	V = np.zeros((n, n), dtype=complex)
	vertices = set(range(n))

	for i in range(n):
		for j in range(i + 1, n):
			total = 0.0 + 0.0j
			for R in itertools.combinations(vertices - {i, j}, ell - 1):
				S_i = tuple(sorted(R + (i,)))
				S_j = tuple(sorted(R + (j,)))

				if v_top_arr is not None:
					# Array-backed lookup
					try:
						idx_i = subset_index[S_i]  # type: ignore[index]
						idx_j = subset_index[S_j]  # type: ignore[index]
					except KeyError:
						continue
					total += v_top_arr[idx_i].conjugate() * v_top_arr[idx_j]
				else:
					# Mapping-backed lookup
					val_i = v_top_map.get(S_i, 0.0)  # type: ignore[union-attr]
					val_j = v_top_map.get(S_j, 0.0)  # type: ignore[union-attr]
					total += np.conj(val_i) * val_j

			V[i, j] = 0.5 * total
			V[j, i] = np.conj(V[i, j])

	if logger is not None:
		try:
			logger.info(
				"Built voting matrix V (n=%d, ell=%d, dtype=%s) in %.3fs",
				int(n),
				int(ell),
				str(V.dtype),
				time.perf_counter() - t0,
			)
		except Exception:
			pass

	return V

