parameters {
  unit_vector[3] direction;
  cov_matrix[3] covariance;
  corr_matrix[3] correlation;
  cholesky_factor_cov[4, 3] covariance_factor;
  cholesky_factor_corr[3] correlation_factor;
}
model {
  target += 0 * (sum(direction) + sum(covariance) + sum(correlation)
                 + sum(covariance_factor) + sum(correlation_factor));
}
