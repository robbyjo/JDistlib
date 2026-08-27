data { array[3] real y; }
parameters { vector<lower=-2, upper=2>[3] theta; }
model {
  theta ~ normal(0, 1);
  y ~ normal(theta, 1);
  target += normal_lpdf(y | theta, 1);
}
