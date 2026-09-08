"""Independent NonCentralChiSquareDensityAccuracyTest fixtures (mpmath 1.4.1).

The modified-Bessel density formula is independent of the implementation's
mode-centred Poisson/chi-square recurrence. Some R 4.6.1 dchisq results disagree
with these values because R truncates by absolute error or substitutes a
central approximation after underflow. No R outputs are used as truth here.
Run: python generate-ncchisq-density-high-precision.py
"""
import mpmath as mp

mp.mp.dps = 100
CASES = [
    ("10000", "1", "10"),
    ("10000", "0", "10"),
    ("1e-200", "10", "10"),
    ("1000", "1", "10"),
    ("17", "4.5", "80"),
    ("1", "0", "10"),
    ("1e-100", "5", "2000"),
    ("1", "1000", "2000"),
]
for values in CASES:
    x, df, ncp = map(mp.mpf, values)
    result = (-(x + ncp) / 2 + (df / 4 - mp.mpf(".5")) * mp.log(x / ncp)
              + mp.log(mp.besseli(df / 2 - 1, mp.sqrt(ncp * x))) - mp.log(2))
    print(*values, mp.nstr(result, 70))

# Direct incomplete-gamma mixture, without Poisson-mass-only truncation.
for value in ("1711.55384277344", "1736.00391113281", "1760.45397949219", "1800"):
    x = mp.mpf(value)
    terms = [mp.exp(-6) * mp.mpf(6) ** k / mp.factorial(k)
             * mp.gammainc(mp.mpf("2.25") + k, x / 2, mp.inf, regularized=True)
             for k in range(300)]
    result = mp.fsum(terms)
    assert terms[-1] < result * mp.mpf("1e-90")
    print("upper_log_cdf", value, mp.nstr(mp.log(result), 70))

for values in (("2", "1e16", "10000"), ("1", "1e12", "1e12"),
               ("2", "1e12", "1e6"), ("1e10", "1e21", "1e10")):
    x, size, mean = map(mp.mpf, values)
    result = (mp.loggamma(size + x) - mp.loggamma(size) - mp.loggamma(x + 1)
              - size * mp.log1p(mean / size) + x * (mp.log(mean) - mp.log(size + mean)))
    print("negative_binomial_log_density", *values, mp.nstr(result, 70))
