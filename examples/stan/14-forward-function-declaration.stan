functions {
  real regularizer(real x);
  real regularizer(real x) { return square(x); }
}
parameters { real theta; }
model {
  target += -0.5 * regularizer(theta);
  theta ~ normal(0, 1);
}
