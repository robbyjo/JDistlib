parameters { vector[3] beta; }
model {
  vector[2] prediction = csr_matrix_times_vector(2,3,[1,2,3]',{1,3,2},{1,3,4},beta);
  matrix[2,3] dense = csr_to_dense_matrix(2,3,[1,2,3]',{1,3,2},{1,3,4});
  target += sum(prediction) + sum(csr_extract_w(dense)) - dot_self(beta);
}
