from enum import Enum
from typing import Any

import numpy as np
from loguru import logger

from lib.plotting_util import tex
from lib.util import make_symmetrical


class HessianNormalization(Enum):
    UNIT = "unit"
    UNIT_DIM = "unit_dim"
    UNIT_DIVIDED_BY_DIM = "unit_divided_by_dim"
    UNIT_DIVIDED_BY_DIM_ROOT = "unit_divided_by_dim_root"
    DIM_ROOT = "dim_root"
    ADAPTIVE = "adaptive"

    def to_plot_label(self):
        base = "||{B_0}|| = "

        match self:
            case HessianNormalization.UNIT:
                return tex(base + "1")
            case HessianNormalization.UNIT_DIM:
                return tex(base + "d")
            case HessianNormalization.UNIT_DIVIDED_BY_DIM:
                return tex(base + "1/d")
            case HessianNormalization.UNIT_DIVIDED_BY_DIM_ROOT:
                return tex(base + "1/\\sqrt{d}")
            case HessianNormalization.DIM_ROOT:
                return tex(base + "\\sqrt{d}")
            case HessianNormalization.ADAPTIVE:
                return base + "adaptive"

    def normalize(self, mat: np.ndarray, **kwargs: dict[str, Any]):
        match self:
            case HessianNormalization.UNIT:
                return mat / np.linalg.norm(mat)

            case HessianNormalization.UNIT_DIM:
                return mat / np.linalg.norm(mat) * mat.shape[0]

            case HessianNormalization.UNIT_DIVIDED_BY_DIM:
                return mat / (np.linalg.norm(mat) * mat.shape[0])

            case HessianNormalization.UNIT_DIVIDED_BY_DIM_ROOT:
                return mat / (np.linalg.norm(mat) * np.sqrt(mat.shape[0]))

            case HessianNormalization.DIM_ROOT:
                return mat / np.linalg.norm(mat) * np.sqrt(mat.shape[0])

            case HessianNormalization.ADAPTIVE:
                if "prev_norm" not in kwargs:
                    raise ValueError(
                        "Previous inv hess norm not passed to adaptive hessian norm"
                    )
                norm = kwargs["prev_norm"]
                if norm is None:
                    logger.warn(
                        "prev_norm set to None in adaptive normalization, ensure this happens only during the initial iteration"
                    )
                    return mat
                return mat / np.linalg.norm(mat) * norm

    def normalize_and_make_symmetrical(self, mat: np.ndarray, **kwargs: dict[str, Any]):
        return make_symmetrical(self.normalize(mat, **kwargs))

    @staticmethod
    def non_degenerate_choices():
        return [
            norm
            for norm in HessianNormalization
            if norm != HessianNormalization.UNIT_DIM
            and norm != HessianNormalization.UNIT_DIVIDED_BY_DIM
        ]
