functions {
  real polynomial_sum(real x, int degree) {
    if (degree == 0) return 1;
    return pow(x, degree) + polynomial_sum(x, degree - 1);
  }
}
parameters { real theta; }
model {
  target += -0.01 * polynomial_sum(theta, 4);
  theta ~ normal(0, 1);
}
