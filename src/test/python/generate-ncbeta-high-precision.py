"""Generate NonCentralBetaAccuracyTest references with mpmath 1.4.1."""

import mpmath as mp

# Enough precision to form complements of the subnormal-tail cases below.
mp.mp.dps = 400


def tail(a, b, ncp, x, lower):
    a, b, mean, x = map(mp.mpf, (a, b, mp.mpf(ncp) / 2, x))
    k = int(mp.floor(mean))
    weight = mp.exp(-mean + k * mp.log(mean) - mp.loggamma(k + 1))

    def beta_tail(i):
        if lower:
            return mp.betainc(a + i, b, 0, x, regularized=True)
        return mp.betainc(b, a + i, 0, 1 - x, regularized=True)

    result = weight * beta_tail(k)
    w = weight
    for i in range(k - 1, -1, -1):
        w *= (i + 1) / mean
        result += w * beta_tail(i)
    w = weight
    i = k
    while True:
        i += 1
        w *= mean / i
        term = w * beta_tail(i)
        result += term
        if term < result * mp.mpf("1e-95") and i > k + 30:
            return result


cases = {
    "ordinary": ("2.5", "7", "11", ".3"),
    "large_ncp": ("5", ".5", "1000", ".8"),
    "small_lower": (".25", "40", "200", ".01"),
    "underflow": ("1.0149511645559415", "126.03373615036026",
                  "1370.9880095911913", "1e-40"),
    "small_upper": ("3.009369292533997", "187.95519621829496",
                    "37.405600013184028", ".847507562537541"),
}

for name, args in cases.items():
    raw_lower = tail(*args, True)
    raw_upper = tail(*args, False)
    lower = 1 - raw_upper if raw_lower > mp.mpf(".5") else raw_lower
    upper = 1 - raw_lower if raw_upper > mp.mpf(".5") else raw_upper
    print(name, "lower", mp.nstr(mp.log(lower), 70))
    print(name, "upper", mp.nstr(mp.log(upper), 70))

low, high = mp.mpf(0), mp.mpf(1)
for _ in range(250):
    mid = (low + high) / 2
    if tail("2.5", "7", "11", mid, True) < mp.mpf(".91"):
        low = mid
    else:
        high = mid
print("q91", mp.nstr((low + high) / 2, 70))

for df2 in ("1000000", "10000000", "100000000"):
    for f_value in (1 / mp.pi, mp.mpf(1), mp.pi):
        y = mp.mpf(5) * f_value / (mp.mpf(df2) + mp.mpf(5) * f_value)
        print("pnf", df2, mp.nstr(f_value, 20),
              mp.nstr(tail("2.5", str(mp.mpf(df2) / 2), "1", str(y), True), 70))

y = mp.mpf("4.5") * 8 / (11 + mp.mpf("4.5") * 8)
print("pnf-upper-log", mp.nstr(mp.log(tail("2.25", "5.5", "7", str(y), False)), 70))
