data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
}
model {
  alpha ~ normal(0, 5);
  beta ~ normal(0, 5);
  y ~ normal(alpha + beta * x, 1);
}
generated quantities {
  real y_at_zero = normal_rng(alpha, 1);
}
