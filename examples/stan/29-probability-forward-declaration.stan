functions {
  real robust_lpdf(real y, real location);
  real robust_lpdf(real y, real location) {
    return student_t_lpdf(y | 5, location, 1);
  }
}
parameters { real location; }
model {
  0.25 ~ robust(location);
  location ~ normal(0, 2);
}
