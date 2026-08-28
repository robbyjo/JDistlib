parameters { vector[3] x; real a; }
model {
  matrix[2,2] L = [[exp(a),0],[a,2]];
  matrix[2,2] K = multiply_lower_tri_self_transpose(L);
  target += mean(x) + variance(x) + sd(x+[0,1,2]') + log_sum_exp(x);
  target += distance(x,[1,2,4]') + squared_distance(x,[1,2,4]');
  target += log_determinant_spd(K) + trace(inverse_spd(K)) - dot_self(x) - square(a);
}
