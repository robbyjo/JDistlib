# Historical SourceForge ticket audit for 0.6.0

This audit replaces the old practice of leaving JDistLib tickets open merely to
monitor R Bugzilla. R's historical individual Bugzilla URLs are no longer a
stable monitoring mechanism. For 0.6.0, behavior is checked against the tagged
R 4.6.1 sources in `.upstream/R-4.6.1`, and relevant cases have focused Java
regressions in `SourceForgeTicketRegressionTest`.

| Ticket | Subject | 0.6.0 disposition |
| --- | --- | --- |
| [#42](https://sourceforge.net/p/jdistlib/tickets/42/) | Maxwell--Boltzmann distribution | Implemented in 0.5.0 as `MaxwellBoltzmann`; closed. |
| [#39](https://sourceforge.net/p/jdistlib/tickets/39/) | Public integration methods | Implemented in 0.5.0; closed. |
| [#30](https://sourceforge.net/p/jdistlib/tickets/30/) | R PR#16845, noncentral-t CDF | Current R 4.6.1 `pnt.c` behavior is present and covered by both the R 4.6.1 audit and a focused upper-log-tail regression. Ready to close. |
| [#23](https://sourceforge.net/p/jdistlib/tickets/23/) | R PR#16332, beta CDF | The difficult `pbeta` log-tail case is covered by a focused regression. A proposed R source/test patch is in `contrib/r-nmath-pbeta-pr16332`. Ready to close. |
| [#15](https://sourceforge.net/p/jdistlib/tickets/15/) | R PR#1662/#7801, Fisher exact test | Not applicable: JDistLib does not implement `stats::fisher.test`, which is outside R's `src/nmath` distribution library. Close as out of scope. |
| [#13](https://sourceforge.net/p/jdistlib/tickets/13/) | Noncentral-beta density precision | The reported value agrees with the high-precision reference and now has a focused regression. Ready to close. |
| [#12](https://sourceforge.net/p/jdistlib/tickets/12/) | R PR#15554, huge-order Bessel functions | `Bessel.j` and `Bessel.y` now reject orders above the reliable recurrence range before integer conversion or allocation. Ready to close. |
| [#11](https://sourceforge.net/p/jdistlib/tickets/11/) | R PR#15628, Poisson density precision | The reported huge-mean, near-mean density agrees with the corrected reference and now has a focused regression. Ready to close. |
| [#9](https://sourceforge.net/p/jdistlib/tickets/9/) | R PR#15635, noncentral chi-square precision | The mode-centred/log-scale implementation keeps the reported tiny tail finite; the historical suite and a focused regression cover it. Ready to close. |
| [#5](https://sourceforge.net/p/jdistlib/tickets/5/) | R PR#7393, hypergeometric integer inputs | Current integer-tolerance and noninteger-support behavior is present and now covered directly. Ready to close. |
| [#1](https://sourceforge.net/p/jdistlib/tickets/1/) | R PR#8528, gamma CDF | Huge-parameter lower log tails remain finite and are covered directly. Ready to close. |

Closing or commenting on the external tracker is intentionally a separate
maintainer action. Suggested closure text for tickets #1, #5, #9, #11, #12,
#13, #23, and #30 is: “Resolved in JDistLib 0.6.0; see
`SourceForgeTicketRegressionTest` and `SOURCEFORGE_TICKET_AUDIT.md`.” For #15,
use: “Closed as out of scope: this tracks `stats::fisher.test`, which JDistLib
does not implement.”

Future upstream monitoring should use tagged R release sources and R's `NEWS`
records, with an executable regression for every adopted numerical fix. It
should not use a permanently open JDistLib ticket as a bookmark.
