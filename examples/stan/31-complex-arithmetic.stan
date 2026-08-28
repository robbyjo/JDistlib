parameters { complex z; }
model {
  complex centered = z - (1+2i);
  target += -norm(centered) + get_real(exp(z) + sin(z));
}
