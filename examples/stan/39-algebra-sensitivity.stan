functions {
  vector equation(vector y, vector theta, array[] real xr, array[] int xi) {
    return [square(y[1])-theta[1]]';
  }
}
transformed data { array[1] real xr={0}; array[1] int xi={0}; }
parameters { vector<lower=.1>[1] theta; }
model {
  vector[1] root = algebra_solver_newton(equation,[1]',theta,xr,xi);
  target += normal_lpdf(root[1] | 1.5,.2);
  theta ~ lognormal(0,1);
}
