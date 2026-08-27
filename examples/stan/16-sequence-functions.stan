transformed data {
  vector[4] x = [3, 1, 4, 2]';
  vector[4] cumulative = cumulative_sum(x);
  vector[4] ascending = sort_asc(x);
  vector[4] descending = sort_desc(x);
  vector[4] backwards = reverse(x);
  vector[2] middle = segment(x, 2, 2);
}
parameters { real theta; }
model {
  target += 0 * (sum(cumulative) + sum(ascending) + sum(descending)
                 + sum(backwards) + sum(middle));
  theta ~ normal(0, 1);
}
