#include <Rmath.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void check(const char *name, double actual, double expected,
		  double tolerance)
{
    double error = fabs(actual - expected);
    printf("%-24s %.17g  error %.3g\n", name, actual, error);
    if (!(error <= tolerance)) {
	fprintf(stderr, "%s exceeded tolerance %.3g\n", name, tolerance);
	exit(1);
    }
}

int main(void)
{
    check("moderate upper log",
	  pnbeta(.3, 2.5, 7., 11., 0, 1),
	  -0.076653268744653658856, 5e-15);
    check("extreme upper log",
	  pnbeta(.847507562537541, 3.009369292533997,
		  187.95519621829496, 37.405600013184028, 0, 1),
	  -255.3388790702448814, 2e-12);
    check("underflow lower log",
	  pnbeta(1e-40, 1.0149511645559415, 126.03373615036026,
		  1370.9880095911913, 1, 1),
	  -774.0719328515007845, 2e-12);
    check("underflow log quantile",
	  qnbeta(-774.0719328515007845, 1.0149511645559415,
		  126.03373615036026, 1370.9880095911913, 1, 1),
	  1e-40, 1e-52);
    return 0;
}
