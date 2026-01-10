from enum import Enum

import numpy as np

from lib.util import make_symmetrical


class HessianNormalization(Enum):
    UNIT = "unit"
    UNIT_DIM = "unit_dim"
    UNIT_DIVIDED_BY_DIM = "unit_divided_by_dim"
    UNIT_DIVIDED_BY_DIM_ROOT = "unit_divided_by_dim_root"

    def to_plot_label(self):
        match self:
            case HessianNormalization.UNIT:
                return "skalowanie jednostkowe ($\\sqrt{\\sum_{ij}{C_{ij}^2}} = 1$)"
            case HessianNormalization.UNIT_DIM:
                return "skalowanie jednostkowe do wymiarowości ($\\sqrt{\\sum{ij}{C_{ij}^2}} = d$)"

    def normalize(self, mat: np.ndarray):
        match self:
            case HessianNormalization.UNIT:
                return mat / np.linalg.norm(mat)

            case HessianNormalization.UNIT_DIM:
                return mat / np.linalg.norm(mat) * mat.shape[0]

            case HessianNormalization.UNIT_DIVIDED_BY_DIM:
                return mat / (np.linalg.norm(mat) * mat.shape[0])

            case HessianNormalization.UNIT_DIVIDED_BY_DIM_ROOT:
                return mat / (np.linalg.norm(mat) * np.sqrt(mat.shape[0]))

    def normalize_and_make_symmetrical(self, mat: np.ndarray):
        return make_symmetrical(self.normalize(mat))
