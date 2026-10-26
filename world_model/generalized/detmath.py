"""Platform-independent exp and log.

``math.exp`` and ``math.log`` call the C library, and C libraries differ
in the last bit (glibc, MSVC and macOS each round some inputs their own
way). Anything that ends up in a consensus hash (the canonical world
checkpoint, the federated mint) must not depend on which machine
computed it, so those paths use these instead.

Both are built only from IEEE 754 basic operations (+, -, *, /) plus
``frexp`` / ``ldexp``, which are exact, so every platform computes the
same bits. Accuracy is within a couple of ulp of the true value.
"""

from __future__ import annotations

import math

# ln 2 split so that k * _LN2_HI is exact for |k| < 2**11 (fdlibm).
_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_INV_LN2 = 1.44269504088896338700e+00
_SQRT_HALF = 0.70710678118654752440

_EXP_MAX = 709.782712893384
_EXP_MIN = -745.1332191019411

# 1/n! for n = 13..0, Horner order.
_EXP_COEFFS = tuple(1.0 / math.factorial(n) for n in range(13, -1, -1))
# 1/(2n+1) for n = 11..1, Horner order in s**2.
_LOG_COEFFS = tuple(1.0 / (2 * n + 1) for n in range(11, 0, -1))


def exp(x: float) -> float:
    x = float(x)
    if x != x:
        return x
    if x > _EXP_MAX:
        raise OverflowError("math range error")
    if x < _EXP_MIN:
        return 0.0
    k = int(round(x * _INV_LN2))
    r = (x - k * _LN2_HI) - k * _LN2_LO     # |r| <= ~0.347
    p = 0.0
    for c in _EXP_COEFFS:
        p = p * r + c
    return math.ldexp(p, k)


def log(x: float) -> float:
    x = float(x)
    if x != x:
        return x
    if x <= 0.0:
        raise ValueError("math domain error")
    if x == math.inf:
        return x
    m, e = math.frexp(x)                    # x = m * 2**e, m in [0.5, 1)
    if m < _SQRT_HALF:
        m *= 2.0
        e -= 1
    s = (m - 1.0) / (m + 1.0)               # |s| <= ~0.1716
    s2 = s * s
    p = 0.0
    for c in _LOG_COEFFS:
        p = p * s2 + c
    log_m = 2.0 * s * (1.0 + s2 * p)
    return e * _LN2_HI + (e * _LN2_LO + log_m)


def log1p(x: float) -> float:
    x = float(x)
    u = 1.0 + x
    if u == 1.0:
        return x
    # Corrects the rounding of 1 + x (Goldberg / Kahan).
    return log(u) * (x / (u - 1.0))
