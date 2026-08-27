data {
  int<lower=1> N;
  array[N] real y;
}
parameters { real mu; }
model {
  mu ~ normal(0, 5);
  y ~ normal(mu, 1);
}
