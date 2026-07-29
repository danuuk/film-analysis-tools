"""The tidy sample table and its cohort selectors.

Sample once, select many times. A cohort is a boolean mask over a shared table, not a bespoke
extraction pipeline — which is the difference between comparing a hypothesis on skin, foliage
and neutrals in minutes and spending half a day reshaping data to fit.

Storage is a dict of named columns (struct-of-arrays), which is what the existing packs
already are: ~260k rows of ~17 columns, about 12 MB. At that scale masking is instant, so the
representation is an ergonomic choice rather than a performance one, and
``dict[str, np.ndarray]`` converts to a dataframe in one line if group-by work ever justifies
the dependency.

Columns not stored on disk are derived on demand from RGB, so ``hue_sector`` works on a pack
that never recorded it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.colour import features
from film_analysis_tools.core.errors import SelectionError

RGB_COLUMN = "rgb_display_linear_l100"

OPERATORS = ("gt", "ge", "lt", "le", "ne", "in", "between")


@dataclass(frozen=True)
class SampleTable:
    """Named columns of equal length, plus the cohort expression that produced them."""

    columns: Mapping[str, np.ndarray]
    name: str = "samples"
    cohort: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        lengths = {int(value.shape[0]) for value in self.columns.values()}
        if len(lengths) > 1:
            raise SelectionError(f"columns have differing lengths: {sorted(lengths)}")

    # ------------------------------------------------------------------ basics

    def __len__(self) -> int:
        for value in self.columns.values():
            return int(value.shape[0])
        return 0

    def __iter__(self) -> Iterator[str]:
        return iter(self.columns)

    @property
    def column_names(self) -> list[str]:
        return sorted(self.columns)

    @property
    def rgb(self) -> np.ndarray:
        return self.column(RGB_COLUMN)

    @property
    def label(self) -> str:
        """A human-readable name including the cohort expression, for reports."""
        return f"{self.name}[{' & '.join(self.cohort)}]" if self.cohort else self.name

    def column(self, name: str) -> np.ndarray:
        """A stored column, or one derived from RGB on demand."""
        if name in self.columns:
            return self.columns[name]
        derived = self._derive(name)
        if derived is None:
            raise SelectionError(
                f"unknown column {name!r}; available: {self.column_names} "
                f"(derivable: {sorted(self._derivable())})"
            )
        return derived

    # ------------------------------------------------------------- derivation

    def _derivable(self) -> set[str]:
        if RGB_COLUMN not in self.columns:
            return set()
        return set(features.feature_columns(np.zeros((1, 3), dtype=np.float64)))

    def _derive(self, name: str) -> np.ndarray | None:
        if RGB_COLUMN not in self.columns:
            return None
        computed = features.feature_columns(self.columns[RGB_COLUMN])
        return computed.get(name)

    # --------------------------------------------------------------- selection

    def select(self, mask: np.ndarray, *, describe: str = "") -> SampleTable:
        """A new table holding the rows where ``mask`` is true."""
        flags = np.asarray(mask, dtype=bool)
        if flags.shape[0] != len(self):
            raise SelectionError(f"mask length {flags.shape[0]} does not match {len(self)} rows")
        return replace(
            self,
            columns={key: value[flags] for key, value in self.columns.items()},
            cohort=(*self.cohort, describe) if describe else self.cohort,
        )

    def where(self, **predicates: Any) -> SampleTable:
        """Select rows matching every predicate.

        Predicates are ``column=value`` for equality, or ``column__op=value`` where ``op`` is
        one of ``gt``, ``ge``, ``lt``, ``le``, ``ne``, ``in``, ``between``.
        """
        if not predicates:
            return self
        mask = np.ones(len(self), dtype=bool)
        described: list[str] = []
        for key, value in sorted(predicates.items()):
            column_name, operator = _split_predicate(key)
            mask &= _apply(self.column(column_name), operator, value, key)
            described.append(_describe(column_name, operator, value))
        return self.select(mask, describe=" & ".join(described))

    def counts(self, column: str) -> dict[str, int]:
        """Row counts per distinct value — for checking a cohort is not empty or tiny."""
        values, totals = np.unique(self.column(column), return_counts=True)
        return {str(value): int(total) for value, total in zip(values, totals, strict=True)}


def _split_predicate(key: str) -> tuple[str, str]:
    if "__" in key:
        column_name, _, operator = key.rpartition("__")
        if operator in OPERATORS:
            return column_name, operator
    return key, "eq"


def _apply(column: np.ndarray, operator: str, value: Any, key: str) -> np.ndarray:
    if operator == "eq":
        return np.asarray(column == value)
    if operator == "ne":
        return np.asarray(column != value)
    if operator == "gt":
        return np.asarray(column > value)
    if operator == "ge":
        return np.asarray(column >= value)
    if operator == "lt":
        return np.asarray(column < value)
    if operator == "le":
        return np.asarray(column <= value)
    if operator == "in":
        return np.isin(column, np.asarray(list(value)))
    if operator == "between":
        low, high = value
        return np.asarray((column >= low) & (column < high))
    raise SelectionError(f"unsupported operator in {key!r}: {operator!r}")


def _describe(column_name: str, operator: str, value: Any) -> str:
    symbols = {"eq": "==", "ne": "!=", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}
    if operator in symbols:
        return f"{column_name} {symbols[operator]} {value!r}"
    if operator == "in":
        return f"{column_name} in {list(value)!r}"
    return f"{column_name} in [{value[0]!r}, {value[1]!r})"


__all__ = ["OPERATORS", "RGB_COLUMN", "SampleTable"]
