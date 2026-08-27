functions {
  vector center(vector x, real location) {
    return x - location;
  }
  real custom_lpdf(vector y, real location) {
    return normal_lpdf(y | location, 1);
  }
  real regularize_lp(real x) {
    target += normal_lpdf(x | 0, 2);
    return 0;
  }
}
data {
  vector[3] y;
}
parameters {
  real mu;
}
model {
  center(y, mu) ~ custom(0);
  target += regularize_lp(mu);
}
