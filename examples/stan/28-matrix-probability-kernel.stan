transformed data {
  vector[2] y = [.2, -.1]';
  matrix[2,2] Sigma = [[1,.2],[.2,1.5]];
}
parameters { vector[2] mu; }
model {
  mu ~ normal(0, 2);
  y ~ multi_normal(mu, Sigma);
}
