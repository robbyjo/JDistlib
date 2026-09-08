/* Copyright (C) 2026 Roby Joehanes; GPL-3.0-or-later */
package jdistlib.evd;

/** Stable conversions shared by the extreme-value distributions. */
final class TailMath {
    private TailMath() {}
    static double logComplement(double logP) {
        return logP > -Math.log(2.0) ? Math.log(-Math.expm1(logP))
                : Math.log1p(-Math.exp(logP));
    }
    static double probability(double logTail, boolean sameTail, boolean logP) {
        if (logP) return sameTail ? logTail : logComplement(logTail);
        return sameTail ? Math.exp(logTail) : -Math.expm1(logTail);
    }
    static double logLower(double p, boolean lower, boolean logP) {
        if (Double.isNaN(p) || (logP ? p > 0.0 : p < 0.0 || p > 1.0))
            return Double.NaN;
        return lower ? (logP ? p : Math.log(p))
                : (logP ? logComplement(p) : Math.log1p(-p));
    }
    static boolean invalid(double loc, double scale, double shape) {
        return !Double.isFinite(loc) || !(scale > 0.0)
                || !Double.isFinite(scale) || !Double.isFinite(shape);
    }
}
