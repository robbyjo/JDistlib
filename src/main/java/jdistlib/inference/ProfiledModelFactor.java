/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Model factor instrumented with low-overhead call, time, and non-finite counters. */
public interface ProfiledModelFactor extends ModelFactor { FactorProfile profile(); }
