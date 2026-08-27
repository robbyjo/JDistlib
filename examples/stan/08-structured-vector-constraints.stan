parameters {
  positive_ordered[3] cutpoints;
  sum_to_zero_vector[3] effects;
}
model {
  cutpoints ~ normal(0, 3);
  effects ~ normal(0, 1);
}
