data {
  array[3] real x;
  array[3] real y;
}
parameters { real alpha; real beta; }
model {
  alpha ~ normal(0, 2);
  beta ~ normal(0, 1);
  y ~ normal(alpha + beta .* x, 1);
  target += -0.001 * sum(square(to_vector(x)));
  target += -0.001 * dot_product(to_vector(x), to_vector(x));
}
