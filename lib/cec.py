import cecxx


def get_cec2017_for_dim(idx: int, dim: int):
    if idx < 1 or idx > 30:
        raise ValueError("invalid idx for cec2017 fun")

    return cecxx.get_cec_function(
        cecxx.CECEdition.CEC2017, idx, dim, subtract_y_global=True
    )
