/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference;

/** Optional named inclusion indicators for a reversible-jump model family. */
public interface ReversibleJumpInclusionTarget extends ReversibleJumpTarget {
    String[] candidateNames();
    boolean active(long modelId, int candidate);
}
