transformed data {
  array[2] vector[2] points = {[1,2]', [3,4]'};
}
parameters { vector[2] beta; }
model {
  beta ~ normal(0, 1);
  target += 0.01 * (dot_product(points[1], beta) + dot_product(points[2], beta));
}
