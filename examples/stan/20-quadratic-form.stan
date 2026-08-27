transformed data {
  matrix[2, 2] precision = [[2, .5], [.5, 3]];
  vector[2] x = [1, -1]';
  real energy = quad_form(precision, x);
}
parameters { real theta; }
model {
  target += 0 * energy;
  theta ~ normal(0, 1);
}
