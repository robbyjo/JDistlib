functions {
  real kernel(real x, real xc, array[] real theta, array[] real xr, array[] int xi) {
    return xr[1]*exp(-theta[1]*x);
  }
}
transformed data { array[1] real xr={2}; array[1] int xi={0}; }
parameters { array[1] real<lower=.1> theta; }
model {
  real area = integrate_1d(kernel,0,1,theta,xr,xi,1e-8);
  target += normal_lpdf(area | 1.2,.2);
  theta ~ lognormal(0,1);
}
