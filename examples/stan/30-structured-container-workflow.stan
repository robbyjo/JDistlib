parameters {
  unit_vector[3] direction;
  corr_matrix[3] correlation;
  positive_ordered[3] cutpoints;
}
transformed parameters {
  matrix[3,3] scaled = diag_pre_multiply(cutpoints, correlation);
  vector[3] projected = scaled * direction;
}
model {
  target += -0.01 * dot_product(projected, projected);
}
