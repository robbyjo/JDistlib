/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import jdistlib.math.IntegrationOptions;
import jdistlib.math.UnivariateFunction;

/** Isolates direct analyzer probes when the integration settings request it. */
final class CallbackProbeExecutor implements UnivariateFunction, AutoCloseable {
	private final UnivariateFunction delegate;
	private final IntegrationOptions options;
	private final long startedNanos = System.nanoTime();
	private final ExecutorService worker;
	private CallbackLimitException terminalFailure;

	CallbackProbeExecutor(UnivariateFunction delegate, IntegrationOptions options) {
		this.delegate = delegate;
		this.options = options;
		worker = options.getCallbackExecution()
				== IntegrationOptions.CallbackExecution.ISOLATED_DAEMON
				? Executors.newSingleThreadExecutor(new ThreadFactory() {
					@Override public Thread newThread(Runnable task) {
						Thread thread = new Thread(task,
								"jdistlib-isolated-diagnostic-probe");
						thread.setDaemon(true);
						return thread;
					}
				}) : null;
	}

	@Override public double eval(final double x) {
		if (terminalFailure != null) throw terminalFailure;
		if (worker == null) return delegate.eval(x);
		long remaining = remainingTotalNanos();
		if (remaining <= 0L) throw fail("total callback time limit exceeded", x);
		Future<Double> future = worker.submit(new Callable<Double>() {
			@Override public Double call() { return delegate.eval(x); }
		});
		try {
			return future.get(Math.min(remaining, options.getMaxCallbackNanos()),
					TimeUnit.NANOSECONDS);
		} catch (TimeoutException exception) {
			future.cancel(true);
			throw fail("isolated callback probe exceeded its time limit", x);
		} catch (InterruptedException exception) {
			future.cancel(true);
			Thread.currentThread().interrupt();
			throw fail("interrupted while waiting for isolated callback probe", x);
		} catch (ExecutionException exception) {
			Throwable original = exception.getCause();
			if (original instanceof RuntimeException) {
				throw (RuntimeException) original;
			}
			if (original instanceof Error) throw (Error) original;
			throw new RuntimeException(original);
		}
	}

	private long remainingTotalNanos() {
		long limit = options.getMaxTotalNanos();
		if (limit == Long.MAX_VALUE) return Long.MAX_VALUE;
		return limit - (System.nanoTime() - startedNanos);
	}

	private CallbackLimitException fail(String message, double x) {
		terminalFailure = new CallbackLimitException(message + " at x=" + x);
		return terminalFailure;
	}

	@Override public void close() {
		if (worker != null) worker.shutdownNow();
	}

	/** Distinguishes resource limits from ordinary callback exceptions. */
	static final class CallbackLimitException extends RuntimeException {
		private static final long serialVersionUID = 1L;
		CallbackLimitException(String message) { super(message); }
	}
}
