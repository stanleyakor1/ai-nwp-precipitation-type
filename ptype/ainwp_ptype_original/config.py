

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PTYPE_CLASSES = ("rain", "snow", "ice pellets", "freezing rain")
PTYPE_SHORT_NAMES = ("rain", "snow", "icep", "frzr")
PROFILE_VARS = ("T_HEIGHT", "DPT_HEIGHT", "U_HEIGHT", "V_HEIGHT")
EXPECTED_HEIGHTS = np.arange(0, 5001, 250, dtype=np.float32)
N_HEIGHTS = len(EXPECTED_HEIGHTS)
N_FEATURES = len(PROFILE_VARS) * N_HEIGHTS


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str
    output_dir: Path
    surface_geopotential_path: Path
    target_heights: np.ndarray = field(default_factory=lambda: EXPECTED_HEIGHTS.copy())
    output_uncertainties: bool = True
    overwrite: bool = False

    def __post_init__(self):
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "surface_geopotential_path", Path(self.surface_geopotential_path))
        object.__setattr__(self, "target_heights", np.asarray(self.target_heights, dtype=np.float32))


def required_output_vars(output_uncertainties=True):
    required = set(PROFILE_VARS) | {
        "ML_rain", "ML_snow", "ML_icep", "ML_frzr",
        "ML_crain", "ML_csnow", "ML_cicep", "ML_cfrzr", "ML_ptype",
        "BOURG_ptype", "BOURG_PA", "BOURG_NA",
        "BOURG_crain", "BOURG_csnow", "BOURG_cicep", "BOURG_cfrzr",
    }
    if output_uncertainties:
        required.add("ML_u")
    return required
