transformed data {
  matrix[3,2] X = [[1,0], [1,1], [1,2]];
  matrix[3,2] Q = qr_thin_Q(X);
  matrix[2,2] R = qr_thin_R(X);
  matrix[2,2] gram = crossprod(X);
  matrix[2,2] L = cholesky_decompose(gram);
}
parameters { real theta; }
model {
  target += 0 * (sum(Q) + sum(R) + sum(L));
  theta ~ normal(0, 1);
}
