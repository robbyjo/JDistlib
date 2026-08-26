/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

/** Ranked family fits and the selected successful candidate. */
public final class CopulaSelectionResult {
	private final CopulaSelectionCriterion criterion;
	private final List<CopulaFitResult> rankings;
	private final CopulaFitResult selected;

	CopulaSelectionResult(CopulaSelectionCriterion criterion,
			List<CopulaFitResult> rankings, CopulaFitResult selected) {
		this.criterion = criterion;
		this.rankings = Collections.unmodifiableList(new ArrayList<>(rankings));
		this.selected = selected;
	}

	public CopulaSelectionCriterion getCriterion() { return criterion; }
	public List<CopulaFitResult> getRankings() { return rankings; }
	public CopulaFitResult getSelected() { return selected; }
	public boolean isSuccess() { return selected != null; }
}
