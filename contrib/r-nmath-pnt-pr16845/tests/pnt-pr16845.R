## R PR#16845: the CDF must not jump at the old |ncp| ~= 37.62 cutoff.
cases <- rbind(
    c(limit=.0378, rsd=3.75,   expected=.15429891097300658974488321321376159),
    c(limit=.0378, rsd=3.76,   expected=.15539778452845913495491252761260947),
    c(limit=.02,   rsd=3.759,  expected=.45187612058802224726627608785540290),
    c(limit=.02,   rsd=3.7591, expected=.45188815299772598689884861170274824))

actual <- apply(cases, 1L, function(z)
    pt(1/z[["limit"]], df=1, ncp=sqrt(2)*100/z[["rsd"]]))
stopifnot(all.equal(actual, cases[, "expected"], tolerance=1e-11),
          abs(actual[2L] - actual[1L]) < .002,
          abs(actual[4L] - actual[3L]) < .00002)
