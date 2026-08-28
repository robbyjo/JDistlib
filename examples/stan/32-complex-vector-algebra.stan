parameters { complex_vector[2] z; }
model {
  complex energy = z' * z;
  target += -get_real(energy) + get_real(sum(conj(z)));
}
