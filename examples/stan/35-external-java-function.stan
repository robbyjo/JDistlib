functions { real java_penalty(real x, real scale); }
parameters { real x; real scale; }
model {
  target += -java_penalty(x,scale);
  x ~ normal(0,1);
  scale ~ normal(1,1);
}
