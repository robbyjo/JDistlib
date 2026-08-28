/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.finance;

import jdistlib.math.Complex;

/** Distribution exposing stable log characteristic and cumulant transforms. */
public interface TransformDistribution {
	Complex logCharacteristic(double frequency);
	Complex logMomentGenerating(double argument);
	TransformDomain momentGeneratingDomain();
}
