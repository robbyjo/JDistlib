parameters { vector[3] logits; }
transformed parameters {
  vector[3] probabilities = softmax(logits);
  vector[3] log_probabilities = log_softmax(logits);
}
model {
  logits ~ normal(0, 1);
  target += 0.1 * sum(log_probabilities) + 0 * sum(probabilities);
}
