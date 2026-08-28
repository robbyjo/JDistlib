functions {
  tuple(real, vector) scale(tuple(real, vector) x, real a) {
    return (a*x.1, a*x.2);
  }
  real total(tuple(real, vector) x) { return x.1 + sum(x.2); }
}
parameters { real x; }
model {
  tuple(real, vector[2]) input = (x, [x,2*x]');
  target += total(scale(input,2)) - square(x);
}
