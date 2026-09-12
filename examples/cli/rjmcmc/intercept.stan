data {
  int<lower=1> N;
  matrix[N,3] X;
  vector[N] y;
}
parameters {
  real alpha;

}
model {
  target += normal_lpdf(alpha | 0, 5);

  for (n in 1:N)
    target += normal_lpdf(y[n] | alpha, 0.75);
}
