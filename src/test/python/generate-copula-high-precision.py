"""Print independent high-precision mixed and vine copula reference rows.

The script uses only Python's Decimal implementation.  Formulas are evaluated
directly from the Clayton copula rather than through JDistlib.  Vine density
factorization follows Aas, Czado, Frigessi, and Bakken (2009),
doi:10.1016/j.insmatheco.2007.02.001.
"""

from decimal import Decimal, getcontext

getcontext().prec = 90
D = Decimal


def power(value, exponent):
    return (exponent * value.ln()).exp()


def clayton_cdf(coordinates, theta):
    if any(value == 0 for value in coordinates):
        return D(0)
    total = sum(power(value, -theta) for value in coordinates) - D(len(coordinates) - 1)
    return power(total, -D(1) / theta)


def clayton_h(conditioning, target, theta):
    if target == 0:
        return D(0)
    if target == 1:
        return D(1)
    total = power(conditioning, -theta) + power(target, -theta) - D(1)
    return power(conditioning, -theta - D(1)) * power(
        total, -D(1) / theta - D(1))


def clayton_log_density(first, second, theta):
    total = power(first, -theta) + power(second, -theta) - D(1)
    return ((D(1) + theta).ln()
            + (-D(1) - theta) * (first.ln() + second.ln())
            + (-D(2) - D(1) / theta) * total.ln())


def mixed_bernoulli(u, probability, outcome, theta):
    split = D(1) - probability
    lower, upper = (D(0), split) if outcome == 0 else (split, D(1))
    return clayton_h(u, upper, theta) - clayton_h(u, lower, theta)


def discrete_bernoulli(first_probability, first_outcome,
                       second_probability, second_outcome, theta):
    first_split = D(1) - first_probability
    second_split = D(1) - second_probability
    first_lower, first_upper = ((D(0), first_split) if first_outcome == 0
                                else (first_split, D(1)))
    second_lower, second_upper = ((D(0), second_split) if second_outcome == 0
                                  else (second_split, D(1)))
    return (clayton_cdf([first_upper, second_upper], theta)
            - clayton_cdf([first_lower, second_upper], theta)
            - clayton_cdf([first_upper, second_lower], theta)
            + clayton_cdf([first_lower, second_lower], theta))


def c_vine_log_density(point, theta01, theta02, theta12_given0):
    conditioned1 = clayton_h(point[0], point[1], theta01)
    conditioned2 = clayton_h(point[0], point[2], theta02)
    return (clayton_log_density(point[0], point[1], theta01)
            + clayton_log_density(point[0], point[2], theta02)
            + clayton_log_density(conditioned1, conditioned2,
                                   theta12_given0))


def d_vine_log_density(point, theta01, theta12, theta02_given1):
    conditioned0 = clayton_h(point[1], point[0], theta01)
    conditioned2 = clayton_h(point[1], point[2], theta12)
    return (clayton_log_density(point[0], point[1], theta01)
            + clayton_log_density(point[1], point[2], theta12)
            + clayton_log_density(conditioned0, conditioned2,
                                   theta02_given1))


MIXED_CASES = [
    ("mixed_center_zero", "mixed", D("1.7"), D("0.37"), D("0.28"), 0, None, None),
    ("mixed_center_one", "mixed", D("1.7"), D("0.37"), D("0.28"), 1, None, None),
    ("mixed_lower_tail", "mixed", D("2.4"), D("0.001"), D("0.02"), 1, None, None),
    ("mixed_upper_tail", "mixed", D("0.8"), D("0.999999"), D("0.98"), 0, None, None),
    ("discrete_rare_00", "discrete", D("2.2"), None, D("0.02"), 0, D("0.03"), 0),
    ("discrete_rare_11", "discrete", D("2.2"), None, D("0.02"), 1, D("0.03"), 1),
    ("discrete_opposing", "discrete", D("0.65"), None, D("0.97"), 1, D("0.04"), 0),
]

VINE_CASES = [
    ("c_center", "C_VINE", [D("0.17"), D("0.63"), D("0.91")],
     D("0.8"), D("2.1"), D("1.4")),
    ("c_lower_boundary", "C_VINE", [D("0.000001"), D("0.000002"), D("0.04")],
     D("1.2"), D("0.7"), D("2.5")),
    ("c_opposing_boundaries", "C_VINE", [D("0.999999"), D("0.00001"), D("0.83")],
     D("0.55"), D("1.8"), D("0.9")),
    ("d_center", "D_VINE", [D("0.12"), D("0.54"), D("0.87")],
     D("1.1"), D("2.2"), D("0.6")),
    ("d_boundary", "D_VINE", [D("0.0001"), D("0.999"), D("0.02")],
     D("0.75"), D("1.6"), D("2.3")),
]


print("MIXED")
print("id,kind,theta,continuous_u,first_probability,first_outcome,"
      "second_probability,second_outcome,expected")
for case in MIXED_CASES:
    name, kind, theta, u, first_p, first_y, second_p, second_y = case
    expected = (mixed_bernoulli(u, first_p, first_y, theta)
                if kind == "mixed" else
                discrete_bernoulli(first_p, first_y, second_p, second_y, theta))
    print(",".join("" if value is None else str(value) for value in
                   (name, kind, theta, u, first_p, first_y,
                    second_p, second_y, expected)))

print("VINE")
print("id,structure,u0,u1,u2,theta0,theta1,theta2,expected_log_density")
for name, structure, point, theta0, theta1, theta2 in VINE_CASES:
    expected = (c_vine_log_density(point, theta0, theta1, theta2)
                if structure == "C_VINE" else
                d_vine_log_density(point, theta0, theta1, theta2))
    print(",".join(str(value) for value in
                   (name, structure, *point, theta0, theta1, theta2, expected)))
