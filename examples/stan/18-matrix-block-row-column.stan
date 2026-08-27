transformed data {
  matrix[3, 3] A = [[1,2,3], [4,5,6], [7,8,9]];
  matrix[2, 2] center = block(A, 1, 2, 2, 2);
  row_vector[3] second_row = row(A, 2);
  vector[3] third_column = col(A, 3);
}
parameters { real theta; }
model {
  target += 0 * (sum(center) + sum(second_row) + sum(third_column));
  theta ~ normal(0, 1);
}
