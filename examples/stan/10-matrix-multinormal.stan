data {
  matrix[3, 2] X;
  vector[3] y;
  matrix[3, 3] Sigma;
}
parameters {
  vector[2] beta;
}
transformed parameters {
  vector[3] mu = X * beta;
}
model {
  beta ~ normal(0, 2);
  y ~ multi_normal(mu, Sigma);
  target += 0 * log_determinant(Sigma);
  target += 0 * sum(mdivide_left_spd(Sigma, y));
  target += 0 * (sum(qr_thin_Q(X)) + sum(qr_thin_R(X)));
}
