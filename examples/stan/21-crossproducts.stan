transformed data {
  matrix[3, 2] X = [[1,0], [1,1], [1,2]];
  matrix[2, 2] gram = crossprod(X);
  matrix[3, 3] row_gram = tcrossprod(X);
  row_vector[2] column_norms = columns_dot_product(X, X);
  vector[3] row_norms = rows_dot_product(X, X);
}
parameters { real theta; }
model {
  target += 0 * (sum(gram) + sum(row_gram) + sum(column_norms) + sum(row_norms));
  theta ~ normal(0, 1);
}
