transformed data {
  matrix[2, 2] A = [[1,2], [3,4]];
  vector[2] scale = [2,3]';
  matrix[2, 2] left_scaled = diag_pre_multiply(scale, A);
  matrix[2, 2] right_scaled = diag_post_multiply(scale, A);
}
parameters { real theta; }
model {
  target += 0 * (sum(left_scaled) + sum(right_scaled));
  theta ~ normal(0, 1);
}
