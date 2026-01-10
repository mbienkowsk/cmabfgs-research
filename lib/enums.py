from enum import Enum

import numpy as np


class HessianNormalization(Enum):
    UNIT = "unit"
    UNIT_DIM = "unit_dim"

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
                dimensions = mat.shape[0]
                return mat / np.linalg.norm(mat) * dimensions
