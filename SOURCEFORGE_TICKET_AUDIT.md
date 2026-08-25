# Historical SourceForge ticket audit for 0.6.0

This audit replaces the old practice of leaving JDistLib tickets open merely to
monitor R Bugzilla. R's historical individual Bugzilla URLs are no longer a
stable monitoring mechanism. The unresolved or historically significant items
below have been migrated to GitHub with their original references, exact
reproduction status, current results, and acceptance criteria. Focused tests in
`SourceForgeTicketRegressionTest` are identified as regressions only for the
behavior they actually exercise; they are not treated as proof that a broader
historical report is resolved.

| Ticket | GitHub | Subject | 0.6.0 disposition |
| --- | --- | --- | --- |
| [#42](https://sourceforge.net/p/jdistlib/tickets/42/) | -- | Maxwell--Boltzmann distribution | Implemented in 0.5.0 as `MaxwellBoltzmann`; closed. |
| [#39](https://sourceforge.net/p/jdistlib/tickets/39/) | -- | Public integration methods | Implemented in 0.5.0; closed. |
| [#30](https://sourceforge.net/p/jdistlib/tickets/30/) | [#8](https://github.com/robbyjo/JDistlib/issues/8) | R PR#16845, noncentral-t CDF | Open. The archived R examples exactly reproduce the discontinuity at the abrupt normal-approximation switch near `ncp = 37.62`. The earlier moderate upper-log-tail test is unrelated to that defect. |
| [#23](https://sourceforge.net/p/jdistlib/tickets/23/) | -- | R PR#16332, beta CDF | The difficult `pbeta` log-tail case is covered by a focused regression. A proposed R source/test patch is in `contrib/r-nmath-pbeta-pr16332`. |
| [#15](https://sourceforge.net/p/jdistlib/tickets/15/) | [#7](https://github.com/robbyjo/JDistlib/issues/7) | R PR#1662/#7801, Fisher exact test | Closed as not planned. JDistLib does not implement `stats::fisher.test`; R fixed the FEXACT workspace problem in R 3.5.0. |
| [#13](https://sourceforge.net/p/jdistlib/tickets/13/) | [#6](https://github.com/robbyjo/JDistlib/issues/6) | Noncentral-beta density precision | Open. Current relative error is about `1.43e-14`: substantially improved, but not bit-for-bit agreement with the reference. Investigate compensated or double-double summation. |
| [#12](https://sourceforge.net/p/jdistlib/tickets/12/) | [#5](https://github.com/robbyjo/JDistlib/issues/5) | R PR#15554, huge-order Bessel functions | Closed as completed. `Bessel.j` and `Bessel.y` reject unrepresentable orders before integer conversion or allocation, and the exact `2^64` cases return `NaN`. |
| [#11](https://sourceforge.net/p/jdistlib/tickets/11/) | [#4](https://github.com/robbyjo/JDistlib/issues/4) | R PR#15628, Poisson density precision | Open. The reported point passes, but JDistLib still uses `bd0()` and has not ported R's `ebd0()`/newer `dpois_raw()` formulation. |
| [#9](https://sourceforge.net/p/jdistlib/tickets/9/) | [#3](https://github.com/robbyjo/JDistlib/issues/3) | R PR#15635, noncentral chi-square precision | Open. The five original lower-tail CDF/quantile round trips still reproduce. Direct upper-tail inversion is much more accurate; the tiny log tail remains finite. |
| [#5](https://sourceforge.net/p/jdistlib/tickets/5/) | [#2](https://github.com/robbyjo/JDistlib/issues/2) | R PR#7393, hypergeometric integer inputs | Open. Density rejects noninteger population parameters while CDF, quantile, and RNG paths round them. The earlier test covers only noninteger support values, not this inconsistency. |
| [#1](https://sourceforge.net/p/jdistlib/tickets/1/) | [#1](https://github.com/robbyjo/JDistlib/issues/1) | R PR#8528, gamma CDF | Closed as completed. The historical call returns an unavoidable ordinary-scale zero and a finite log probability rather than `NaN`. |

Closing or commenting on the SourceForge tracker remains a separate maintainer
action. SourceForge tickets should link to their corresponding GitHub issue;
they should not be described as resolved unless the GitHub issue is closed with
the exact historical reproduction covered. GitHub issues #2, #3, #4, #6, and
#8 remain open for 0.6.0 work. Issues #1 and #5 record completed fixes, and #7
records the out-of-scope `fisher.test` disposition.

Future upstream monitoring should use tagged R release sources and R's `NEWS`
records, with an executable regression for every adopted numerical fix. It
should not use a permanently open JDistLib ticket as a bookmark.
