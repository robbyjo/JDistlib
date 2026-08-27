transformed data {
  vector[2] a = [1, 2]';
  vector[2] b = [3, 4]';
  vector[4] vertical = append_row(a, b);
  matrix[2, 2] columns = append_col(a, b);
  array[4] real values = append_array({1, 2}, {3, 4});
}
parameters { real theta; }
model {
  target += 0 * (sum(vertical) + sum(columns) + sum(values));
  theta ~ normal(0, 1);
}
