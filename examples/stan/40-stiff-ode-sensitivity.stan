functions {
  vector decay(real t, vector y, vector theta, array[] real xr, array[] int xi) {
    return [-theta[1]*y[1]]';
  }
}
transformed data { array[2] real ts={.1,.2}; array[1] real xr={0}; array[1] int xi={0}; }
parameters { vector<lower=.1>[1] theta; }
model {
  array[2] vector[1] trajectory = ode_bdf(decay,[1]',0,ts,theta,xr,xi);
  target += normal_lpdf(trajectory[2,1] | .8,.1);
  theta ~ lognormal(0,1);
}
