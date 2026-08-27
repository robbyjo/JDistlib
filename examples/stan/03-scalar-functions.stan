functions {
  real centered(real x, real location) {
    return x - location;
  }
  real soft_penalty(real x) {
    if (x > 0) return square(x);
    return square(x) / 2;
  }
  real magnitude(real x) {
    return x >= 0 ? x : -x;
  }
}
data { real y; }
parameters { real mu; }
model {
  target += normal_lpdf(centered(y, mu) | 0, 1);
  target += -0.1 * soft_penalty(mu);
  target += 0 * magnitude(mu);
}
