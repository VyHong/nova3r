import math
from numbers import Real


def resolve_eval_freq(eval_freq):
    """Convert an epoch interval into Lightning's validation settings.

    Fractions in (0, 1) validate within each epoch; positive whole numbers
    validate every N epochs. Return (check_val_every_n_epoch, val_check_interval).
    """
    if (
        isinstance(eval_freq, bool)
        or not isinstance(eval_freq, Real)
        or not math.isfinite(eval_freq)
        or eval_freq <= 0
    ):
        raise ValueError("eval_freq must be a positive finite number of epochs")
    if eval_freq < 1:
        return 1, float(eval_freq)
    if int(eval_freq) != eval_freq:
        raise ValueError("eval_freq must be a fraction below 1 or a positive whole number")
    return int(eval_freq), 1.0
