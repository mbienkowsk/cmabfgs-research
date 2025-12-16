from enum import Enum


class CECProvider(Enum):
    opfunu = "opfunu"
    cecxx = "cecxx"


def get_cec2017_for_dim(idx: int, dim: int, provider: CECProvider = CECProvider.cecxx):
    """opfunu left for debugging/comparison"""
    min, max = (1, 29) if provider == CECProvider.opfunu else (1, 30)
    if idx < min or idx > max:
        raise ValueError(
            f"invalid idx for cec2017 fun from provider {provider.value}: {idx}"
        )

    if provider == CECProvider.cecxx:
        import cecxx

        return cecxx.get_cec_function(
            cecxx.CECEdition.CEC2017, idx, dim, subtract_y_global=True
        )
    else:
        from opfunu.cec_based import cec2017

        fname = f"F{idx}2017"
        fn = getattr(cec2017, fname)(dim)
        return fn.evaluate
