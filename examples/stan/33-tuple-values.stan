parameters { real x; }
model {
  tuple(real, tuple(vector[2], complex)) state = (x, ([x,2*x]', 1+2i));
  state.2.1 = [2*x,3*x]';
  target += state.1 + sum(state.2.1) + get_real(state.2.2) - square(x);
}
