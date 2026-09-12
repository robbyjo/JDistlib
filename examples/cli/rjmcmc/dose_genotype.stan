data {
  int<lower=1> N;
  matrix[N,3] X;
  vector[N] y;
}
parameters {
  real alpha;
  real dose;
  real genotype;
}
model {
  target += normal_lpdf(alpha | 0, 5);
  target += normal_lpdf(dose | 0, 2);
  target += normal_lpdf(genotype | 0, 2);
  for (n in 1:N)
    target += normal_lpdf(y[n] | alpha + dose * X[n,1] + genotype * X[n,2], 0.75);
}
