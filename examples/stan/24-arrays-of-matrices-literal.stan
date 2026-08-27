transformed data {
  array[2] matrix[2,2] transforms = {[[1,0],[0,1]], [[2,0],[0,2]]};
}
parameters { vector[2] beta; }
model {
  beta ~ normal(0, 1);
  target += 0.01 * sum(transforms[2] * beta);
}
