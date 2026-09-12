data {
  int<lower=1> N;
  matrix[N,3] X;
  vector[N] y;
}
parameters {
  real alpha;
  real genotype;
  real proxy_marker;
}
model {
  target += normal_lpdf(alpha | 0, 5);
  target += normal_lpdf(genotype | 0, 2);
  target += normal_lpdf(proxy_marker | 0, 2);
  for (n in 1:N)
    target += normal_lpdf(y[n] | alpha + genotype * X[n,2] + proxy_marker * X[n,3], 0.75);
}
