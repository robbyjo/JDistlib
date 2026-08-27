transformed data {
  vector[4] x = [1,2,3,4]';
  vector[2] first = head(x, 2);
  vector[2] last = tail(x, 2);
  matrix[4, 3] repeated_column = rep_matrix(x, 3);
  matrix[2, 4] repeated_row = rep_matrix([1,2,3,4], 2);
}
parameters { real theta; }
model {
  target += 0 * (sum(first) + sum(last) + sum(repeated_column) + sum(repeated_row));
  theta ~ normal(0, 1);
}
