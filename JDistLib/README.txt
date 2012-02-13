This program is a manual Java translation of R's distribution library
found in src/nmath folder of the source tar ball. The translation took
place from February 10, 2012 to February 12, 2012 and was based on
R version 2.14.1. The original code was in C, so the translation is
relatively straightforward, except in a few routines that are peppered
with gotos.

What distributions are included? Virtually all standard distribution
included in R. In alphabetical order:
1. Beta
2. Binomial
3. Cauchy
4. Chi square
5. Exponential
6. Fisher's F
7. Gamma
8. Geometric
9. Hypergeometric
10. Logistic
11. Log normal
12. Negative binomial
13. Noncentral beta
14. Noncentral chi square
15. Noncentral F
16. Noncentral T
17. Normal
18. Poisson
19. Sign Rank
20. Student's T
21. Tukey
22. Uniform
23. Weibull
24. Wilcoxon

Each of these distributions has density (pdf), cumulative (cdf),
quantile, and random number generation (RNG) routines associated with
it, except for Tukey's distribution, which only has cdf and quantile
routines. These routines are implemented as static final functions
for ease of use, except for Sign Rank and Wilcoxon distributions.
Sign Rank and Wilcoxon distributions are implemented as dynamic
classes since they require a storage matrix that is dependent on the
supplied parameters.

The primary change I did with the source code is I made the routines
thread safe, especially for the RNG routine for each distribution.
In R, the RNGs of several distributions require some global state
variables, which hinders the implementation of thread safety. I think
this is why R is not a multi-threaded program. Multi-threaded
libraries in R, such as multicore, got around this by forking
processes (i.e., copy the entire memory used by R for each cores).
This results in a huge memory waste since multi-threaded R program
will consume k times more than it ought to be, where k is the number
of cores being used. I got around this by implementing some structures
to afford state storage or eliminating the requirement altogether.
For Beta and Gamma RNG routines, the time saved by storing the states
is meager. So, I eliminated the states for these distributions. In
Binomial, Hypergeometric, and Poisson distributions, I implemented
the state storage as an inner class that has to be instantiated upon
use. Since Sign Rank and Wilcoxon distributions are implemented as
dynamic classes, storing the states is as simple as declaring the
states as fields. 

There was an earlier attempt called distlib, which was based on an
earlier version of R. However, the library currently suffers a few
shortcomings:
1. It was based on an earlier version of R. The newer distribution
library has been updated to improve accuracy. Gamma distribution has
been significantly improved. Since many other key distributions use
routines in the Gamma distribution, their accuracy is also markedly
improved, especially in the extreme lower tail.
2. Distlib is buggy and cannot even compile presently.
3. Distlib is not thread safe due to the global states required by
some RNG routines.
4. Distlib was a result of an automatic translation. The resulting
code is very messy.

This is why I decided to do the translation over.

Known problems:
1. Java does not have "long double". Hence I changed every
occurrence of "long double" into "double". This happens in
Hypergeometric, Noncentral beta, Noncentral chi square, Noncentral T,
and Tukey distributions. Possible ramifications: loss of precision in
these distributions. See the "TODO long double" tag in each of the
file.
2. R authors noted a precision problem in the quantile routine of
the Hypergeometric distribution and have not fixed it. The problem is
most pronounced at the very extreme tail of the distribution.
I translated the file as such. So, the resulting translation will also
suffer from the same problem. In addition to that, further precision
loss should be expected due to the "long double" problem above.
3. I did only minimal testing. So, caveat emptor.

 
February 12, 2012
Roby Joehanes
