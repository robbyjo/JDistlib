data {
  matrix[2, 2] x;
  array[2] real y;
}
parameters { vector[2] beta; }
model {
  beta ~ normal(0, 2);
  for (n in 1:2)
    y[n] ~ normal(x[n, 1] * beta[1] + x[n, 2] * beta[2], 1);
}
