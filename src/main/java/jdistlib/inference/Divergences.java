/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Extracts sampler pathologies with coordinates suitable for plotting. */
public final class Divergences {
	private Divergences() {}
	public static List<DivergenceLocation> locate(ChainResult chain, BayesianModel model) {
		List<DivergenceLocation> result = new ArrayList<DivergenceLocation>();
		for (int i = 0; i < chain.size(); i++) if (chain.statisticsAt(i).divergent()) {
			double[] unconstrained = chain.sample(i);
			result.add(new DivergenceLocation(i, unconstrained,
					model == null ? null : model.constrain(unconstrained),
					chain.statisticsAt(i).energyError()));
		}
		return Collections.unmodifiableList(result);
	}
}
