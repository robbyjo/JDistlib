data {
  int<lower=1> N;
  matrix[N,3] X;
  vector[N] y;
}
parameters {
  real alpha;
  real genotype;
}
model {
  target += normal_lpdf(alpha | 0, 5);
  target += normal_lpdf(genotype | 0, 2);
  for (n in 1:N)
    target += normal_lpdf(y[n] | alpha + genotype * X[n,2], 0.75);
}
