# Integration reference corpus

The adjacent CSV is deliberately independent of JDistlib's quadrature code.
Its decimal targets were evaluated to at least 70 significant digits from
closed-form identities using arbitrary-precision decimal/MPFR arithmetic:

* `oscillatory_sine_squared`: integral of `sin(100 x)^2` on `[0, pi]` is
  `pi / 2`.
* `endpoint_beta_half`: the beta integral `B(1/2, 1/2)` is `pi`.
* `interior_inverse_square_root`: splitting at `a = 0.3` gives
  `2 (sqrt(a) + sqrt(1-a))`.
* `large_scaled_polynomial`: integral of `10^200 x^20` is `10^200 / 21`.
* `narrow_triangular_mode`: a unit-height triangle with half-width `10^-6`
  has area `10^-6`.
* `pareto_quarter_tail`: integral of `(1+x)^(-5/4)` on `[0, infinity)` is `4`.

The corpus keeps decimal strings rather than binary-double outputs so future
implementations can test higher-precision numeric types against the same data.
