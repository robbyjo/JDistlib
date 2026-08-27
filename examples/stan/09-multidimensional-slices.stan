data {
  array[2, 3] real x;
}
parameters {
  real theta;
}
model {
  array[2, 3] real work = x;
  work[1, 2:3] = rep_array(theta, 2);
  work[:, 1] += theta;
  target += sum(work);
  theta ~ normal(0, 1);
}
