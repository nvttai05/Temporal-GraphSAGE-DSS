from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureState:
    numeric: dict[str, dict[str, float]]
    categorical: dict[str, tuple[str, ...]]
    multivalue: dict[str, tuple[str, ...]]
    output_columns: tuple[str, ...]


class FeatureBuilder:
    """Train-only fitted numeric scaler and categorical encoder."""

    UNKNOWN = "__UNKNOWN__"
    MISSING = "__MISSING__"

    def __init__(
        self,
        *,
        numeric_columns: Iterable[str] = (),
        categorical_columns: Iterable[str] = (),
        multivalue_columns: Iterable[str] = (),
        multivalue_delimiter: str = "|",
    ):
        self.numeric_columns = tuple(numeric_columns)
        self.categorical_columns = tuple(categorical_columns)
        self.multivalue_columns = tuple(multivalue_columns)
        self.multivalue_delimiter = multivalue_delimiter
        overlap = (
            (set(self.numeric_columns) & set(self.categorical_columns))
            | (set(self.numeric_columns) & set(self.multivalue_columns))
            | (set(self.categorical_columns) & set(self.multivalue_columns))
        )
        if overlap:
            raise ValueError(f"Columns cannot be both numeric and categorical: {overlap}")
        self._state: FeatureState | None = None

    @property
    def fitted(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> FeatureState:
        if self._state is None:
            raise RuntimeError("FeatureBuilder must be fit on train before transform")
        return self._state

    @property
    def state_hash(self) -> str:
        state = self.state
        payload = json.dumps(
            {
                "numeric": state.numeric,
                "categorical": state.categorical,
                "multivalue": state.multivalue,
                "output_columns": state.output_columns,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _require_columns(self, frame: pd.DataFrame) -> None:
        required = {
            "split",
            *self.numeric_columns,
            *self.categorical_columns,
            *self.multivalue_columns,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Feature frame missing columns: {missing}")

    def fit(self, train: pd.DataFrame) -> "FeatureBuilder":
        self._require_columns(train)
        splits = set(train["split"].astype(str).unique())
        if splits != {"train"}:
            raise ValueError(f"FeatureBuilder.fit accepts train only, got splits={splits}")
        numeric: dict[str, dict[str, float]] = {}
        output: list[str] = []
        for column in self.numeric_columns:
            values = pd.to_numeric(train[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            numeric[column] = {"median": median, "mean": mean, "std": std}
            output.append(f"{column}__scaled")
        categorical: dict[str, tuple[str, ...]] = {}
        for column in self.categorical_columns:
            values = train[column].fillna(self.MISSING).astype(str)
            vocabulary = tuple(sorted(set(values)))
            categorical[column] = vocabulary
            output.extend(f"{column}=={value}" for value in vocabulary)
            output.append(f"{column}=={self.UNKNOWN}")
        multivalue: dict[str, tuple[str, ...]] = {}
        for column in self.multivalue_columns:
            token_sets = self._token_sets(train[column])
            vocabulary = tuple(sorted({token for tokens in token_sets for token in tokens}))
            multivalue[column] = vocabulary
            output.extend(f"{column}~={value}" for value in vocabulary)
            output.append(f"{column}~={self.UNKNOWN}")
        self._state = FeatureState(numeric, categorical, multivalue, tuple(output))
        return self

    def _token_sets(self, values: pd.Series) -> list[set[str]]:
        result: list[set[str]] = []
        for value in values:
            if pd.isna(value) or str(value).strip() == "":
                result.append({self.MISSING})
            else:
                result.append(
                    {
                        token.strip()
                        for token in str(value).split(self.multivalue_delimiter)
                        if token.strip()
                    }
                    or {self.MISSING}
                )
        return result

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(frame)
        state = self.state
        output = pd.DataFrame(index=frame.index)
        for column, params in state.numeric.items():
            values = pd.to_numeric(frame[column], errors="coerce").fillna(params["median"])
            output[f"{column}__scaled"] = (values - params["mean"]) / params["std"]
        for column, vocabulary in state.categorical.items():
            values = frame[column].fillna(self.MISSING).astype(str)
            known = set(vocabulary)
            for value in vocabulary:
                output[f"{column}=={value}"] = (values == value).astype(float)
            output[f"{column}=={self.UNKNOWN}"] = (~values.isin(known)).astype(float)
        for column, vocabulary in state.multivalue.items():
            token_sets = self._token_sets(frame[column])
            known = set(vocabulary)
            for value in vocabulary:
                output[f"{column}~={value}"] = [float(value in tokens) for tokens in token_sets]
            output[f"{column}~={self.UNKNOWN}"] = [
                float(bool(tokens - known)) for tokens in token_sets
            ]
        return output.loc[:, state.output_columns].astype(float)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)
