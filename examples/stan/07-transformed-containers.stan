data {
  array[3] real x;
  array[3] real y;
}
transformed data {
  vector[3] centered_x = to_vector(x) - mean(to_vector(x));
}
parameters { real alpha; real beta; }
transformed parameters {
  vector[3] mu = alpha + beta .* centered_x;
}
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1);
  y ~ normal(mu, 1);
}
