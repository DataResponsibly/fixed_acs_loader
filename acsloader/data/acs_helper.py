from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import your forked Folktables classes
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSIncome,
    ACSPublicCoverage,
    ACSTravelTime,
)

# Public mapping users can import if they want
ACS_SCENARIOS: Dict[str, object] = {
    "ACSEmployment": ACSEmployment,
    "ACSIncome": ACSIncome,
    "ACSPublicCoverage": ACSPublicCoverage,
    "ACSTravelTime": ACSTravelTime,
}


@dataclass
class ACSData:
    """
    Wrapper around Folktables to create pandas DataFrames.
    """
    survey_year: Union[int, str] = 2023
    horizon: str = "1-Year"
    survey: str = "person"
    states: Iterable[str] = field(default_factory=lambda: ["CA"])
    use_archive: bool = True
    download: bool = True
    random_state: Optional[int] = None

    # internal store of raw ACS data
    acs_data: Optional[pd.DataFrame] = field(init=False, default=None)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.random_state) if self.random_state is not None else None
        # Keep rng around for reproducible subsamples
        self._rng = rng

        ds = ACSDataSource(
            survey_year=self.survey_year,
            horizon=self.horizon,
            survey=self.survey,
            use_archive=self.use_archive,
        )
        self.acs_data = ds.get_data(states=list(self.states), download=self.download)

    def return_acs_data_scenario(
        self,
        scenario: str = "ACSEmployment",
        subsample: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build DataFrames for a given ACS scenario.

        Returns:
            (all_df, features_df, target_df, group_df)
        """
        if scenario not in ACS_SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {list(ACS_SCENARIOS)}")

        scen = ACS_SCENARIOS[scenario]

        # Extract numpy arrays
        features, label, group = scen.df_to_numpy(self.acs_data)

        if verbose:
            print("features shape:", features.shape, "| label shape:", label.shape, "| group shape:", group.shape)

        # Concatenate features + label for "all" table
        np_all = np.c_[features, label]

        # Optional subsample (without replacement)
        if subsample is not None:
            n = np_all.shape[0]
            k = min(int(subsample), n)
            if self._rng is not None:
                take = self._rng.choice(n, size=k, replace=False)
            else:
                take = np.random.choice(n, size=k, replace=False)
            np_all = np_all[take]
            features = features[take]
            label = label[take]
            group = group[take]

        # Column names from scenario metadata
        feature_cols: List[str] = list(scen._features)
        target_col: str = scen._target
        group_col: str = scen._group

        all_df = pd.DataFrame(np_all, columns=feature_cols + [target_col])
        X_df = pd.DataFrame(features, columns=feature_cols)
        y_df = pd.DataFrame(label, columns=[target_col])

        # Group can be 1-D or 2-D depending on scenario; normalize to 2D
        group_arr = group if group.ndim > 1 else group.reshape(-1, 1)
        group_cols = [group_col] if group_arr.shape[1] == 1 else [f"{group_col}_{i}" for i in range(group_arr.shape[1])]
        g_df = pd.DataFrame(group_arr, columns=group_cols)

        return all_df, X_df, y_df, g_df

    def return_simple_acs_data_scenario(
        self,
        scenario: str = "ACSEmployment",
        subsample: Optional[int] = None,
        verbose: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return only 'simple' (categorical) features plus target and group.
        """
        all_df, X_df, y_df, g_df = self.return_acs_data_scenario(
            scenario=scenario, subsample=subsample, verbose=verbose
        )

        allowed = list(self.get_metadata_features(f_types=[0]).keys())

        # Always include target / group (even if non-categorical)
        for col in list(y_df.columns) + list(g_df.columns):
            if col not in allowed:
                allowed.append(col)

        remove = [c for c in all_df.columns if c not in allowed]

        all_df = all_df.drop(columns=remove, errors="ignore")
        X_df = X_df.drop(columns=remove, errors="ignore")

        return all_df, X_df, y_df, g_df

    def get_acs_names_features(self, verbose: bool = False) -> Dict[str, List[str]]:
        """
        Convenience: scenario name -> feature name list
        """
        out = {}
        for name, scen in ACS_SCENARIOS.items():
            out[name] = list(scen._features)
            if verbose:
                print(name, out[name])
        return out

    def get_metadata_features(self, f_types: Optional[List[int]] = None) -> Dict[str, int]:
        """
        Feature type metadata (hand-curated).

        Codes:
            0: Categorical
            1: Large categorical (>10 categories)
            2: Ordinal
            3: Continuous
        """
        feature_metadata: Dict[str, int] = {
            "AGEP": 2,
            "ANC": 0,
            "CIT": 0,
            "COW": 0,
            "DEAR": 0,
            "DEYE": 0,
            "DIS": 0,
            "DREM": 0,
            "ESP": 0,
            "ESR": 0,
            "FER": 0,
            "JWTR": 1,
            "MAR": 0,
            "MIG": 0,
            "MIL": 0,
            "NATIVITY": 0,
            "OCCP": 1,
            "PINCP": 0,
            "POBP": 1,
            "POVPIP": 3,
            "POWPUMA": 1,
            "PUMA": 1,
            "RAC1P": 0,
            "RELP": 1,
            "SCHL": 1,
            "SEX": 0,
            "ST": 1,
            "WKHP": 2,
            "PUBCOV": 0,
            "JWMNP": 0,
        }
        if f_types is None:
            return feature_metadata

        keys: List[str] = []
        for f_t in f_types:
            keys.extend([k for k, v in feature_metadata.items() if v == f_t])
        return {k: feature_metadata[k] for k in keys if k in feature_metadata}
