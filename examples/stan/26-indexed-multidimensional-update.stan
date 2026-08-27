parameters { real theta; }
model {
  array[2,3] real work = {{1,2,3},{4,5,6}};
  work[1,2:3] = rep_array(theta, 2);
  work[:,1] += theta;
  target += 0.01 * sum(work);
  theta ~ normal(0, 1);
}
