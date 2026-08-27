parameters {
  real<lower=-4> lower_parameter;
  real<upper=4> upper_parameter;
  vector<offset=10, multiplier=2>[2] shifted;
}
model {
  lower_parameter ~ normal(0, 3);
  upper_parameter ~ normal(0, 3);
  shifted ~ normal(10, 3);
}
