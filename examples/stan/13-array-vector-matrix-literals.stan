transformed data {
  array[2, 2] real grid = {{1, 2}, {3, 4}};
  vector[3] column = [1, 2, 3]';
  row_vector[3] row = [4, 5, 6];
  matrix[2, 2] square = [[2, 0], [0, 3]];
}
parameters { real theta; }
model {
  target += 0 * (sum(grid) + sum(column) + sum(row) + sum(square));
  theta ~ normal(0, 1);
}
