/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import static jdistlib.math.MathFunctions.lgammafn;
import static jdistlib.math.MathFunctions.psi;
import static jdistlib.math.PolyGamma.tetragamma;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import jdistlib.Binomial;
import jdistlib.Beta;
import jdistlib.Cauchy;
import jdistlib.ChiSquare;
import jdistlib.Exponential;
import jdistlib.ExponentiallyModifiedGaussian;
import jdistlib.Gamma;
import jdistlib.Geometric;
import jdistlib.HyperGeometric;
import jdistlib.Laplace;
import jdistlib.LogNormal;
import jdistlib.Logistic;
import jdistlib.NegBinomial;
import jdistlib.Normal;
import jdistlib.Poisson;
import jdistlib.T;
import jdistlib.Uniform;
import jdistlib.Weibull;
import jdistlib.inference.BayesianModel;
import jdistlib.inference.Constraints;
import jdistlib.inference.DifferentiableModelFactor;
import jdistlib.inference.ModelBuilder;
import jdistlib.inference.ModelState;
import jdistlib.inference.ParameterConstraint;
import jdistlib.rng.RandomEngine;

/** Parser and reference compiler for the versioned Stan-inspired language subset. */
public final class ModelScript {
	public static final String LANGUAGE_VERSION = "0.8";
	private ModelScript() {}

	public static CompiledModelScript compile(String source, Map<String, double[]> suppliedData) {
		if (source == null || suppliedData == null)
			throw new IllegalArgumentException("source and data are required");
		Parser parser = new Parser(source);
		Program program = parser.parse();
		if (!parser.diagnostics.isEmpty()) throw new ModelScriptException(parser.diagnostics);
		return program.compile(suppliedData);
	}

	public static CompiledModelScript compile(String source) {
		return compile(source, Collections.<String, double[]>emptyMap());
	}

	/** Parses the source without requiring data values or constructing a model. */
	public static void validateSyntax(String source) {
		if (source == null) throw new IllegalArgumentException("source is required");
		Parser parser = new Parser(source);
		parser.parse();
		if (!parser.diagnostics.isEmpty()) throw new ModelScriptException(parser.diagnostics);
	}

	private enum TokenKind { IDENTIFIER, NUMBER, SYMBOL, EOF }
	private static final class Token {
		final TokenKind kind; final String text; final int line; final int column;
		Token(TokenKind kind, String text, int line, int column) {
			this.kind = kind; this.text = text; this.line = line; this.column = column;
		}
	}

	private static final class Lexer {
		final String source; int index; int line = 1; int column = 1;
		Lexer(String source) { this.source = source; }
		Token next() {
			skip();
			if (index >= source.length()) return new Token(TokenKind.EOF, "", line, column);
			int start = index, startLine = line, startColumn = column;
			char c = source.charAt(index);
			if (Character.isLetter(c) || c == '_') {
				advance();
				while (index < source.length()) {
					c = source.charAt(index);
					if (!Character.isLetterOrDigit(c) && c != '_') break;
					advance();
				}
				return new Token(TokenKind.IDENTIFIER, source.substring(start, index), startLine, startColumn);
			}
			if (Character.isDigit(c) || c == '.' && index + 1 < source.length()
					&& Character.isDigit(source.charAt(index + 1))) {
				advance();
				while (index < source.length()) {
					c = source.charAt(index);
					if (!(Character.isDigit(c) || c == '.' || c == 'e' || c == 'E'
							|| (c == '+' || c == '-') && index > start
							&& (source.charAt(index - 1) == 'e' || source.charAt(index - 1) == 'E'))) break;
					advance();
				}
				return new Token(TokenKind.NUMBER, source.substring(start, index), startLine, startColumn);
			}
			if (index + 1 < source.length()) {
				String pair = source.substring(index, index + 2);
				if (pair.equals("+=") || pair.equals("-=") || pair.equals("*=")
						|| pair.equals("/=") || pair.equals("<=") || pair.equals(">=")
						|| pair.equals("==") || pair.equals("!=")
						|| pair.equals("&&") || pair.equals("||")) {
					advance(); advance(); return new Token(TokenKind.SYMBOL, pair, startLine, startColumn);
				}
			}
			advance(); return new Token(TokenKind.SYMBOL, Character.toString(c), startLine, startColumn);
		}
		private void skip() {
			while (index < source.length()) {
				char c = source.charAt(index);
				if (Character.isWhitespace(c)) { advance(); continue; }
				if (c == '/' && index + 1 < source.length() && source.charAt(index + 1) == '/') {
					while (index < source.length() && source.charAt(index) != '\n') advance();
					continue;
				}
				if (c == '/' && index + 1 < source.length() && source.charAt(index + 1) == '*') {
					advance(); advance();
					while (index + 1 < source.length()
							&& !(source.charAt(index) == '*' && source.charAt(index + 1) == '/')) advance();
					if (index + 1 < source.length()) { advance(); advance(); }
					continue;
				}
				break;
			}
		}
		private void advance() {
			if (source.charAt(index++) == '\n') { line++; column = 1; } else column++;
		}
	}

	private static final class Program {
		final List<Declaration> data = new ArrayList<Declaration>();
		final List<Declaration> parameters = new ArrayList<Declaration>();
		final List<Assignment> transformedData = new ArrayList<Assignment>();
		final List<Assignment> transformedParameters = new ArrayList<Assignment>();
		final List<Statement> model = new ArrayList<Statement>();
		final List<Assignment> generated = new ArrayList<Assignment>();

		CompiledModelScript compile(Map<String, double[]> supplied) {
			Map<String, double[]> dataValues = copyData(supplied);
			List<ScriptDiagnostic> diagnostics = new ArrayList<ScriptDiagnostic>();
			Context validationContext = new Context(null, dataValues, 0, null);
			for (Declaration declaration : data) {
				double[] values = dataValues.get(declaration.name);
				if (values == null) diagnostics.add(declaration.error("missing required data '" + declaration.name + "'"));
				else {
					int expected = declaration.dimension == null ? 1
							: checkedDimension(declaration.dimension.eval(validationContext).value, declaration);
					if (values.length != expected) diagnostics.add(declaration.error(
							"data length " + values.length + " does not match declared length " + expected));
					double lower = declaration.lower == null ? Double.NEGATIVE_INFINITY
							: declaration.lower.eval(validationContext).value;
					double upper = declaration.upper == null ? Double.POSITIVE_INFINITY
							: declaration.upper.eval(validationContext).value;
					for (double value : values) {
						if (declaration.integer && value != Math.rint(value))
							diagnostics.add(declaration.error("integer data contains a non-integer value"));
						if (value < lower || value > upper)
							diagnostics.add(declaration.error("data value violates its declared bounds"));
					}
				}
			}
			if (!diagnostics.isEmpty()) throw new ModelScriptException(diagnostics);
			ModelBuilder builder = new ModelBuilder();
			for (Map.Entry<String, double[]> entry : dataValues.entrySet()) builder.data(entry.getKey(), entry.getValue());
			Context constants = new Context(null, dataValues, 0, null);
			for (Assignment assignment : transformedData) {
				Diff value = assignment.expression.eval(constants);
				dataValues.put(assignment.name, new double[] {value.value});
				builder.data(assignment.name, value.value);
				constants.setLocal(assignment.name, new Diff[] {value});
			}
			for (Declaration declaration : parameters) {
				int dimension = declaration.dimension == null ? 1
						: checkedDimension(declaration.dimension.eval(constants).value, declaration);
				ParameterConstraint constraint;
				double[] initial;
				if (declaration.type.equals("simplex")) {
					constraint = Constraints.simplex(dimension); initial = new double[dimension];
					Arrays.fill(initial, 1.0 / dimension);
				} else if (declaration.type.equals("ordered")) {
					constraint = Constraints.ordered(dimension); initial = new double[dimension];
					for (int i = 0; i < dimension; i++) initial[i] = i;
				} else if (declaration.lower != null && declaration.upper != null) {
					double lower = declaration.lower.eval(constants).value;
					double upper = declaration.upper.eval(constants).value;
					if (dimension != 1) throw new ModelScriptException(Collections.singletonList(
							declaration.error("bounded vectors are not yet supported; use scalar declarations")));
					constraint = Constraints.bounded(lower, upper); initial = new double[] {0.5 * (lower + upper)};
				} else if (declaration.lower != null) {
					double lower = declaration.lower.eval(constants).value;
					if (lower != 0.0) throw new ModelScriptException(Collections.singletonList(
							declaration.error("only lower=0 maps to the positive transform")));
					constraint = dimension == 1 ? Constraints.positive() : Constraints.positiveVector(dimension);
					initial = new double[dimension]; Arrays.fill(initial, 1.0);
				} else {
					constraint = dimension == 1 ? Constraints.real() : Constraints.realVector(dimension);
					initial = new double[dimension];
				}
				builder.parameter(declaration.name, constraint, initial);
			}
			final int constrainedDimension = constrainedDimension(parameters, constants);
			final Set<String> declaredNames = new LinkedHashSet<String>();
			for (Declaration declaration : parameters) declaredNames.add(declaration.name);
			declaredNames.addAll(dataValues.keySet());
			boolean procedural = false;
			for (Statement statement : model) procedural |= statement.procedural();
			if (procedural) {
				addModelFactor(builder, "script:model", new BlockStatement(model, false),
						declaredNames, dataValues, constrainedDimension);
			} else {
				int statementIndex = 0;
				for (Statement statement : model)
					addModelFactor(builder, "script:" + (++statementIndex) + ":" + statement.label(),
							statement, declaredNames, dataValues, constrainedDimension);
			}
			final BayesianModel compiled = builder.build();
			final Map<String, double[]> capturedData = copyData(dataValues);
			final List<Assignment> capturedTransforms = new ArrayList<Assignment>(transformedParameters);
			final List<Assignment> capturedGenerated = new ArrayList<Assignment>(generated);
			CompiledModelScript.Generator generator = new CompiledModelScript.Generator() {
				@Override public Map<String, double[]> generate(ModelState state, RandomEngine random) {
					Context context = new Context(state, capturedData, constrainedDimension, random);
					for (Assignment assignment : capturedTransforms)
						context.setLocal(assignment.name, new Diff[] {assignment.expression.eval(context)});
					Map<String, double[]> result = new LinkedHashMap<String, double[]>();
					for (Assignment assignment : capturedGenerated) {
						Diff value = assignment.expression.eval(context);
						context.setLocal(assignment.name, new Diff[] {value});
						result.put(assignment.name, new double[] {value.value});
					}
					return result;
				}
			};
			return new CompiledModelScript(compiled, generator, LANGUAGE_VERSION);
		}

		private void addModelFactor(ModelBuilder builder, String factorName,
				final Statement statement, Set<String> declaredNames,
				Map<String, double[]> dataValues, final int constrainedDimension) {
			Set<String> dependencies = new LinkedHashSet<String>(); statement.collect(dependencies);
			boolean expanded;
			do {
				expanded = false;
				for (Assignment assignment : transformedParameters) {
					if (dependencies.remove(assignment.name)) {
						assignment.expression.collect(dependencies); expanded = true;
					}
				}
			} while (expanded);
			dependencies.retainAll(declaredNames);
			final Map<String, double[]> capturedData = copyData(dataValues);
			final List<Assignment> capturedTransforms =
					new ArrayList<Assignment>(transformedParameters);
			builder.factor(factorName, dependencies.toArray(new String[dependencies.size()]),
					new DifferentiableModelFactor() {
				@Override public double logDensityAndAddGradient(ModelState state,
						double[] gradient) {
					Context context = new Context(state, capturedData, constrainedDimension, null);
					for (Assignment assignment : capturedTransforms)
						context.setLocal(assignment.name,
								new Diff[] {assignment.expression.eval(context)});
					Diff contribution = statement.eval(context);
					for (int i = 0; i < gradient.length; i++) gradient[i] += contribution.gradient[i];
					return contribution.value;
				}
			});
		}
	}

	private static final class Declaration {
		final String type; final String name; final Expr dimension;
		final Expr lower; final Expr upper; final boolean integer; final Token token;
		Declaration(String type, String name, Expr dimension, Expr lower,
				Expr upper, boolean integer, Token token) {
			this.type = type; this.name = name; this.dimension = dimension;
			this.lower = lower; this.upper = upper; this.integer = integer; this.token = token;
		}
		ScriptDiagnostic error(String message) { return new ScriptDiagnostic(token.line, token.column, message); }
	}
	private static final class Assignment {
		final String name; final Expr expression;
		Assignment(String name, Expr expression) { this.name = name; this.expression = expression; }
	}

	private interface Statement {
		Diff eval(Context context);
		void collect(Set<String> names);
		String label();
		default boolean procedural() { return false; }
	}
	private static final class TargetStatement implements Statement {
		final Expr expression;
		TargetStatement(Expr expression) { this.expression = expression; }
		@Override public Diff eval(Context context) { return expression.eval(context); }
		@Override public void collect(Set<String> names) { expression.collect(names); }
		@Override public String label() { return "target"; }
	}
	private static final class BlockStatement implements Statement {
		final List<Statement> statements; final boolean scoped;
		BlockStatement(List<Statement> statements, boolean scoped) {
			this.statements = statements; this.scoped = scoped;
		}
		@Override public Diff eval(Context context) {
			if (scoped) context.pushScope();
			try {
				Diff result = Diff.constant(0.0, context.dimension);
				for (Statement statement : statements) result = result.add(statement.eval(context));
				return result;
			} finally { if (scoped) context.popScope(); }
		}
		@Override public void collect(Set<String> names) {
			for (Statement statement : statements) statement.collect(names);
		}
		@Override public String label() { return "block"; }
		@Override public boolean procedural() {
			if (scoped) return true;
			for (Statement statement : statements) if (statement.procedural()) return true;
			return false;
		}
	}
	private static final class LocalDeclarationStatement implements Statement {
		final String name; final Expr initializer; final boolean integer;
		LocalDeclarationStatement(String name, Expr initializer, boolean integer) {
			this.name = name; this.initializer = initializer; this.integer = integer;
		}
		@Override public Diff eval(Context context) {
			Diff value = initializer.eval(context);
			if (integer && value.value != Math.rint(value.value))
				throw new IllegalArgumentException("integer local '" + name + "' received a non-integer value");
			context.declareLocal(name, new Diff[] {value}, integer);
			return Diff.constant(0.0, context.dimension);
		}
		@Override public void collect(Set<String> names) { initializer.collect(names); }
		@Override public String label() { return "local"; }
		@Override public boolean procedural() { return true; }
	}
	private static final class AssignmentStatement implements Statement {
		final String name; final String operator; final Expr expression;
		AssignmentStatement(String name, String operator, Expr expression) {
			this.name = name; this.operator = operator; this.expression = expression;
		}
		@Override public Diff eval(Context context) {
			Diff value = expression.eval(context);
			if (!operator.equals("=")) {
				Diff current = context.requireScalarLocal(name);
				if (operator.equals("+=")) value = current.add(value);
				else if (operator.equals("-=")) value = current.subtract(value);
				else if (operator.equals("*=")) value = current.multiply(value);
				else if (operator.equals("/=")) value = current.divide(value);
				else throw new IllegalStateException("unknown assignment operator " + operator);
			}
			context.assignLocal(name, new Diff[] {value});
			return Diff.constant(0.0, context.dimension);
		}
		@Override public void collect(Set<String> names) { expression.collect(names); }
		@Override public String label() { return "assign"; }
		@Override public boolean procedural() { return true; }
	}
	private static final class IfStatement implements Statement {
		final Expr condition; final Statement whenTrue; final Statement whenFalse;
		IfStatement(Expr condition, Statement whenTrue, Statement whenFalse) {
			this.condition = condition; this.whenTrue = whenTrue; this.whenFalse = whenFalse;
		}
		@Override public Diff eval(Context context) {
			if (truth(condition.eval(context))) return whenTrue.eval(context);
			return whenFalse == null ? Diff.constant(0.0, context.dimension) : whenFalse.eval(context);
		}
		@Override public void collect(Set<String> names) {
			condition.collect(names); whenTrue.collect(names);
			if (whenFalse != null) whenFalse.collect(names);
		}
		@Override public String label() { return "if"; }
		@Override public boolean procedural() { return true; }
	}
	private static final class ForStatement implements Statement {
		final String variable; final Expr lower; final Expr upper; final Statement body;
		ForStatement(String variable, Expr lower, Expr upper, Statement body) {
			this.variable = variable; this.lower = lower; this.upper = upper; this.body = body;
		}
		@Override public Diff eval(Context context) {
			int from = checkedLoopBound(lower.eval(context).value, variable);
			int to = checkedLoopBound(upper.eval(context).value, variable);
			Diff result = Diff.constant(0.0, context.dimension);
			context.pushScope();
			try {
				context.declareLocal(variable, new Diff[] {Diff.constant(from, context.dimension)}, true);
				for (int value = from; value <= to; value++) {
					context.assignLocal(variable,
							new Diff[] {Diff.constant(value, context.dimension)});
					result = result.add(body.eval(context));
					if (value == Integer.MAX_VALUE) break;
				}
				return result;
			} finally { context.popScope(); }
		}
		@Override public void collect(Set<String> names) {
			lower.collect(names); upper.collect(names); body.collect(names);
		}
		@Override public String label() { return "for"; }
		@Override public boolean procedural() { return true; }
	}
	private static final class WhileStatement implements Statement {
		private static final int MAXIMUM_ITERATIONS = 1000000;
		final Expr condition; final Statement body;
		WhileStatement(Expr condition, Statement body) { this.condition = condition; this.body = body; }
		@Override public Diff eval(Context context) {
			Diff result = Diff.constant(0.0, context.dimension);
			int iterations = 0;
			while (truth(condition.eval(context))) {
				if (++iterations > MAXIMUM_ITERATIONS)
					throw new IllegalStateException("while loop exceeded " + MAXIMUM_ITERATIONS + " iterations");
				result = result.add(body.eval(context));
			}
			return result;
		}
		@Override public void collect(Set<String> names) { condition.collect(names); body.collect(names); }
		@Override public String label() { return "while"; }
		@Override public boolean procedural() { return true; }
	}
	private static final class SamplingStatement implements Statement {
		final Expr left; final String distribution; final List<Expr> arguments;
		SamplingStatement(Expr left, String distribution, List<Expr> arguments) {
			this.left = left; this.distribution = distribution; this.arguments = arguments;
		}
		@Override public Diff eval(Context context) {
			Diff[] observations = left.evalVector(context);
			Diff result = Diff.constant(0.0, context.dimension);
			for (Diff observation : observations) {
				Diff[] values = new Diff[arguments.size() + 1]; values[0] = observation;
				for (int i = 0; i < arguments.size(); i++) values[i + 1] = arguments.get(i).eval(context);
				result = result.add(logProbability(distribution, values));
			}
			return result;
		}
		@Override public void collect(Set<String> names) {
			left.collect(names); for (Expr argument : arguments) argument.collect(names);
		}
		@Override public String label() { return distribution; }
	}

	private interface Expr {
		Diff eval(Context context);
		default Diff[] evalVector(Context context) { return new Diff[] {eval(context)}; }
		void collect(Set<String> names);
	}
	private static final class NumberExpr implements Expr {
		final double value; NumberExpr(double value) { this.value = value; }
		@Override public Diff eval(Context context) { return Diff.constant(value, context.dimension); }
		@Override public void collect(Set<String> names) {}
	}
	private static final class VariableExpr implements Expr {
		final String name; final Expr index;
		VariableExpr(String name, Expr index) { this.name = name; this.index = index; }
		@Override public Diff eval(Context context) {
			Diff[] values = evalVector(context);
			if (values.length != 1) throw new IllegalArgumentException(name + " is a vector; supply an index");
			return values[0];
		}
		@Override public Diff[] evalVector(Context context) {
			Diff[] values = resolveVector(context);
			if (index == null) return values;
			int offset = checkedIndex(index.eval(context).value, values.length, name);
			return new Diff[] {values[offset]};
		}
		private Diff[] resolveVector(Context context) {
			Diff[] local = context.local(name);
			if (local != null) return local;
			if (context.state != null && context.state.hasParameter(name)) {
				int size = context.state.parameterDimension(name);
				Diff[] result = new Diff[size];
				for (int i = 0; i < size; i++) {
					result[i] = Diff.constant(context.state.value(name, i), context.dimension);
					result[i].gradient[context.state.constrainedOffset(name) + i] = 1.0;
				}
				return result;
			}
			double[] values = context.data.get(name);
			if (values == null) throw new IllegalArgumentException("unknown variable: " + name);
			Diff[] result = new Diff[values.length];
			for (int i = 0; i < values.length; i++) result[i] = Diff.constant(values[i], context.dimension);
			return result;
		}
		@Override public void collect(Set<String> names) { names.add(name); if (index != null) index.collect(names); }
	}
	private static final class UnaryExpr implements Expr {
		final String operator; final Expr operand;
		UnaryExpr(String operator, Expr operand) { this.operator = operator; this.operand = operand; }
		@Override public Diff eval(Context context) {
			Diff value = operand.eval(context);
			if (operator.equals("-")) return value.negate();
			if (operator.equals("!")) return booleanValue(!truth(value), context);
			return value;
		}
		@Override public void collect(Set<String> names) { operand.collect(names); }
	}
	private static final class BinaryExpr implements Expr {
		final String operator; final Expr left; final Expr right;
		BinaryExpr(String operator, Expr left, Expr right) {
			this.operator = operator; this.left = left; this.right = right;
		}
		@Override public Diff eval(Context context) {
			Diff a = left.eval(context);
			if (operator.equals("&&") && !truth(a)) return booleanValue(false, context);
			if (operator.equals("||") && truth(a)) return booleanValue(true, context);
			Diff b = right.eval(context);
			if (operator.equals("+")) return a.add(b);
			if (operator.equals("-")) return a.subtract(b);
			if (operator.equals("*")) return a.multiply(b);
			if (operator.equals("/")) return a.divide(b);
			if (operator.equals("^")) return a.pow(b);
			if (operator.equals("<")) return booleanValue(a.value < b.value, context);
			if (operator.equals("<=")) return booleanValue(a.value <= b.value, context);
			if (operator.equals(">")) return booleanValue(a.value > b.value, context);
			if (operator.equals(">=")) return booleanValue(a.value >= b.value, context);
			if (operator.equals("==")) return booleanValue(a.value == b.value, context);
			if (operator.equals("!=")) return booleanValue(a.value != b.value, context);
			if (operator.equals("&&")) return booleanValue(truth(b), context);
			if (operator.equals("||")) return booleanValue(truth(b), context);
			throw new IllegalStateException("unknown operator " + operator);
		}
		@Override public void collect(Set<String> names) { left.collect(names); right.collect(names); }
	}
	private static final class CallExpr implements Expr {
		final String name; final List<Expr> arguments;
		CallExpr(String name, List<Expr> arguments) { this.name = name; this.arguments = arguments; }
		@Override public Diff eval(Context context) {
			Diff[] values = new Diff[arguments.size()];
			for (int i = 0; i < values.length; i++) values[i] = arguments.get(i).eval(context);
			Diff scalar = scalarFunction(name, values, context);
			if (scalar != null) return scalar;
			if (name.endsWith("_lpdf") || name.endsWith("_lpmf")
					|| name.endsWith("_lupdf") || name.endsWith("_lupmf")) {
				int suffixLength = name.endsWith("_lupdf") || name.endsWith("_lupmf") ? 6 : 5;
				String distribution = name.substring(0, name.length() - suffixLength);
				return logProbability(distribution, values);
			}
			if (name.endsWith("_rng")) return random(name.substring(0, name.length() - 4), values, context);
			throw new IllegalArgumentException("unknown function: " + name);
		}
		@Override public void collect(Set<String> names) { for (Expr argument : arguments) argument.collect(names); }
	}

	private static Diff scalarFunction(String name, Diff[] values, Context context) {
		if (name.equals("pi")) return constantFunction(values, name, Math.PI, context);
		if (name.equals("e")) return constantFunction(values, name, Math.E, context);
		if (name.equals("sqrt2")) return constantFunction(values, name, Math.sqrt(2.0), context);
		if (name.equals("log2") && values.length == 0)
			return Diff.constant(Math.log(2.0), context.dimension);
		if (name.equals("log10")) {
			if (values.length == 0) return Diff.constant(Math.log(10.0), context.dimension);
			Diff value = unary(values, name);
			return value.scale(Math.log10(value.value), 1.0 / (value.value * Math.log(10.0)));
		}
		if (name.equals("machine_precision"))
			return constantFunction(values, name, Math.ulp(1.0), context);
		if (name.equals("positive_infinity"))
			return constantFunction(values, name, Double.POSITIVE_INFINITY, context);
		if (name.equals("negative_infinity"))
			return constantFunction(values, name, Double.NEGATIVE_INFINITY, context);
		if (name.equals("not_a_number"))
			return constantFunction(values, name, Double.NaN, context);
		if (name.equals("exp")) return unary(values, name).exp();
		if (name.equals("exp2")) {
			Diff value = unary(values, name); double result = Math.pow(2.0, value.value);
			return value.scale(result, Math.log(2.0) * result);
		}
		if (name.equals("expm1")) return unary(values, name).expm1();
		if (name.equals("log")) return unary(values, name).log();
		if (name.equals("log2")) {
			Diff value = unary(values, name);
			return value.scale(Math.log(value.value) / Math.log(2.0), 1.0 / (value.value * Math.log(2.0)));
		}
		if (name.equals("log1p")) return unary(values, name).log1p();
		if (name.equals("log1m")) {
			Diff value = unary(values, name);
			return value.scale(Math.log1p(-value.value), -1.0 / (1.0 - value.value));
		}
		if (name.equals("log1p_exp")) {
			Diff value = unary(values, name); double result = log1pExp(value.value);
			return value.scale(result, invLogit(value.value));
		}
		if (name.equals("log1m_exp")) {
			Diff value = unary(values, name); double result = log1mExp(value.value);
			return value.scale(result, -Math.exp(value.value) / (1.0 - Math.exp(value.value)));
		}
		if (name.equals("sqrt")) return unary(values, name).sqrt();
		if (name.equals("cbrt")) {
			Diff value = unary(values, name); double result = Math.cbrt(value.value);
			return value.scale(result, 1.0 / (3.0 * result * result));
		}
		if (name.equals("square")) {
			Diff value = unary(values, name); return value.multiply(value);
		}
		if (name.equals("inv")) return unary(values, name).inverse();
		if (name.equals("inv_square")) {
			Diff value = unary(values, name); return value.multiply(value).inverse();
		}
		if (name.equals("inv_sqrt")) return unary(values, name).sqrt().inverse();
		if (name.equals("abs") || name.equals("fabs")) return unary(values, name).abs();
		if (name.equals("floor")) return nondifferentiable(values, name, Math.floor(unary(values, name).value));
		if (name.equals("ceil")) return nondifferentiable(values, name, Math.ceil(unary(values, name).value));
		if (name.equals("round")) return nondifferentiable(values, name, Math.rint(unary(values, name).value));
		if (name.equals("trunc")) {
			Diff value = unary(values, name);
			return nondifferentiable(values, name, value.value < 0.0 ? Math.ceil(value.value) : Math.floor(value.value));
		}
		if (name.equals("sin")) return unary(values, name).sin();
		if (name.equals("cos")) return unary(values, name).cos();
		if (name.equals("tan")) return unary(values, name).tan();
		if (name.equals("sinpi")) {
			Diff value = unary(values, name);
			return value.scale(jdistlib.math.MathFunctions.sinpi(value.value),
					Math.PI * jdistlib.math.MathFunctions.cospi(value.value));
		}
		if (name.equals("cospi")) {
			Diff value = unary(values, name);
			return value.scale(jdistlib.math.MathFunctions.cospi(value.value),
					-Math.PI * jdistlib.math.MathFunctions.sinpi(value.value));
		}
		if (name.equals("tanpi")) {
			Diff value = unary(values, name); double result = jdistlib.math.MathFunctions.tanpi(value.value);
			return value.scale(result, Math.PI * (1.0 + result * result));
		}
		if (name.equals("asin")) return unary(values, name).asin();
		if (name.equals("acos")) return unary(values, name).acos();
		if (name.equals("atan")) return unary(values, name).atan();
		if (name.equals("sinh")) return unary(values, name).sinh();
		if (name.equals("cosh")) return unary(values, name).cosh();
		if (name.equals("tanh")) return unary(values, name).tanh();
		if (name.equals("asinh")) {
			Diff value = unary(values, name);
			return value.scale(Math.log(value.value + Math.sqrt(value.value * value.value + 1.0)),
					1.0 / Math.sqrt(value.value * value.value + 1.0));
		}
		if (name.equals("acosh")) {
			Diff value = unary(values, name);
			return value.scale(Math.log(value.value + Math.sqrt(value.value * value.value - 1.0)),
					1.0 / Math.sqrt(value.value * value.value - 1.0));
		}
		if (name.equals("atanh")) {
			Diff value = unary(values, name);
			return value.scale(0.5 * Math.log((1.0 + value.value) / (1.0 - value.value)),
					1.0 / (1.0 - value.value * value.value));
		}
		if (name.equals("erf")) return unary(values, name).erf();
		if (name.equals("erfc")) return unary(values, name).erfc();
		if (name.equals("tgamma")) return unary(values, name).gamma();
		if (name.equals("lgamma")) return unary(values, name).lgamma();
		if (name.equals("digamma")) return unary(values, name).digamma();
		if (name.equals("trigamma")) return unary(values, name).trigamma();
		if (name.equals("inv_logit")) {
			return scalarInvLogit(unary(values, name));
		}
		if (name.equals("logit")) {
			Diff value = unary(values, name);
			return value.scale(Math.log(value.value / (1.0 - value.value)),
					1.0 / (value.value * (1.0 - value.value)));
		}
		if (name.equals("log_inv_logit")) {
			Diff value = unary(values, name);
			return value.scale(-log1pExp(-value.value), invLogit(-value.value));
		}
		if (name.equals("log1m_inv_logit")) {
			Diff value = unary(values, name);
			return value.scale(-log1pExp(value.value), -invLogit(value.value));
		}
		if (name.equals("Phi")) {
			Diff value = unary(values, name);
			return value.scale(Normal.cumulative(value.value, 0.0, 1.0, true, false),
					Normal.density(value.value, 0.0, 1.0, false));
		}
		if (name.equals("Phi_approx")) {
			Diff value = unary(values, name);
			Diff polynomial = value.multiply(value).multiply(value).multiply(0.07056)
					.add(value.multiply(1.5976));
			double result = invLogit(polynomial.value);
			return polynomial.scale(result, result * (1.0 - result));
		}
		if (name.equals("inv_Phi")) {
			Diff value = unary(values, name);
			double result = Normal.quantile(value.value, 0.0, 1.0, true, false);
			return value.scale(result, 1.0 / Normal.density(result, 0.0, 1.0, false));
		}
		if (name.equals("is_inf")) return predicate(values, name, Double.isInfinite(unary(values, name).value), context);
		if (name.equals("is_nan")) return predicate(values, name, Double.isNaN(unary(values, name).value), context);
		if (name.equals("is_finite")) return predicate(values, name, Double.isFinite(unary(values, name).value), context);
		if (name.equals("step")) return predicate(values, name, unary(values, name).value >= 0.0, context);
		if (name.equals("int_step")) return predicate(values, name, unary(values, name).value > 0.0, context);
		if (name.equals("sign")) return nondifferentiable(values, name, Math.signum(unary(values, name).value));
		if (name.equals("atan2")) {
			Diff[] pair = binary(values, name); Diff y = pair[0], x = pair[1];
			double denominator = x.value * x.value + y.value * y.value;
			return y.combine(Math.atan2(y.value, x.value), x.value / denominator,
					x, -y.value / denominator);
		}
		if (name.equals("pow")) {
			Diff[] pair = binary(values, name); return pair[0].pow(pair[1]);
		}
		if (name.equals("fma")) {
			Diff[] triple = ternary(values, name);
			return triple[0].multiply(triple[1]).add(triple[2]);
		}
		if (name.equals("hypot")) {
			Diff[] pair = binary(values, name); double result = Math.hypot(pair[0].value, pair[1].value);
			return pair[0].combine(result, pair[0].value / result, pair[1], pair[1].value / result);
		}
		if (name.equals("fmin") || name.equals("min")) {
			Diff[] pair = binary(values, name); return pair[0].value <= pair[1].value ? pair[0] : pair[1];
		}
		if (name.equals("fmax") || name.equals("max")) {
			Diff[] pair = binary(values, name); return pair[0].value >= pair[1].value ? pair[0] : pair[1];
		}
		if (name.equals("fdim")) {
			Diff[] pair = binary(values, name);
			return pair[0].value > pair[1].value ? pair[0].subtract(pair[1]) : Diff.constant(0.0, context.dimension);
		}
		if (name.equals("fmod")) {
			Diff[] pair = binary(values, name); double quotient = pair[0].value / pair[1].value;
			double truncated = quotient < 0.0 ? Math.ceil(quotient) : Math.floor(quotient);
			return pair[0].subtract(pair[1].multiply(truncated));
		}
		if (name.equals("log_sum_exp")) {
			Diff[] pair = binary(values, name); double maximum = Math.max(pair[0].value, pair[1].value);
			return Diff.constant(maximum, context.dimension)
					.add(pair[0].subtract(Diff.constant(maximum, context.dimension)).exp()
							.add(pair[1].subtract(Diff.constant(maximum, context.dimension)).exp()).log());
		}
		if (name.equals("log_diff_exp")) {
			Diff[] pair = binary(values, name);
			return pair[0].add(pair[1].subtract(pair[0]).exp().negate().log1p());
		}
		if (name.equals("log_mix")) {
			Diff[] triple = ternary(values, name); Diff theta = triple[0];
			Diff first = theta.log().add(triple[1]);
			Diff second = Diff.constant(1.0, context.dimension).subtract(theta).log().add(triple[2]);
			double maximum = Math.max(first.value, second.value);
			return Diff.constant(maximum, context.dimension).add(first.add(-maximum).exp()
					.add(second.add(-maximum).exp()).log());
		}
		if (name.equals("log_inv_logit_diff")) {
			Diff[] pair = binary(values, name);
			Diff first = scalarInvLogit(pair[0]); Diff second = scalarInvLogit(pair[1]);
			return first.subtract(second).log();
		}
		if (name.equals("multiply_log")) {
			Diff[] pair = binary(values, name); return pair[0].multiply(pair[1].log());
		}
		if (name.equals("lbeta")) {
			Diff[] pair = binary(values, name);
			return pair[0].lgamma().add(pair[1].lgamma()).subtract(pair[0].add(pair[1]).lgamma());
		}
		if (name.equals("binomial_coefficient_log")) {
			Diff[] pair = binary(values, name);
			return pair[0].add(1.0).lgamma().subtract(pair[1].add(1.0).lgamma())
					.subtract(pair[0].subtract(pair[1]).add(1.0).lgamma());
		}
		if (name.equals("log_rising_factorial")) {
			Diff[] pair = binary(values, name);
			return pair[0].add(pair[1]).lgamma().subtract(pair[0].lgamma());
		}
		if (name.equals("log_falling_factorial")) {
			Diff[] pair = binary(values, name);
			return pair[0].add(1.0).lgamma().subtract(pair[0].subtract(pair[1]).add(1.0).lgamma());
		}
		if (name.equals("rising_factorial")) {
			Diff[] pair = binary(values, name);
			return pair[0].add(pair[1]).lgamma().subtract(pair[0].lgamma()).exp();
		}
		if (name.equals("falling_factorial")) {
			Diff[] pair = binary(values, name);
			return pair[0].add(1.0).lgamma()
					.subtract(pair[0].subtract(pair[1]).add(1.0).lgamma()).exp();
		}
		return null;
	}

	private static final class Context {
		final ModelState state; final Map<String, double[]> data; final int dimension;
		final RandomEngine random;
		final List<Map<String, Diff[]>> scopes = new ArrayList<Map<String, Diff[]>>();
		final List<Set<String>> integerScopes = new ArrayList<Set<String>>();
		Context(ModelState state, Map<String, double[]> data, int dimension, RandomEngine random) {
			this.state = state; this.data = data; this.dimension = dimension; this.random = random;
			scopes.add(new LinkedHashMap<String, Diff[]>());
			integerScopes.add(new LinkedHashSet<String>());
		}
		void pushScope() {
			scopes.add(new LinkedHashMap<String, Diff[]>());
			integerScopes.add(new LinkedHashSet<String>());
		}
		void popScope() {
			scopes.remove(scopes.size() - 1); integerScopes.remove(integerScopes.size() - 1);
		}
		Diff[] local(String name) {
			for (int i = scopes.size() - 1; i >= 0; i--) {
				Diff[] value = scopes.get(i).get(name); if (value != null) return value;
			}
			return null;
		}
		void declareLocal(String name, Diff[] value, boolean integer) {
			Map<String, Diff[]> scope = scopes.get(scopes.size() - 1);
			if (scope.containsKey(name)) throw new IllegalArgumentException("duplicate local variable: " + name);
			scope.put(name, value);
			if (integer) integerScopes.get(integerScopes.size() - 1).add(name);
		}
		void setLocal(String name, Diff[] value) { scopes.get(0).put(name, value); }
		void assignLocal(String name, Diff[] value) {
			for (int i = scopes.size() - 1; i >= 0; i--) {
				Map<String, Diff[]> scope = scopes.get(i);
				if (scope.containsKey(name)) {
					if (integerScopes.get(i).contains(name)
							&& (value.length != 1 || value[0].value != Math.rint(value[0].value)))
						throw new IllegalArgumentException("integer local '" + name
								+ "' received a non-integer value");
					scope.put(name, value); return;
				}
			}
			throw new IllegalArgumentException("assignment requires a declared local variable: " + name);
		}
		Diff requireScalarLocal(String name) {
			Diff[] value = local(name);
			if (value == null || value.length != 1)
				throw new IllegalArgumentException("scalar local variable expected: " + name);
			return value[0];
		}
	}
	private static final class Diff {
		final double value; final double[] gradient;
		Diff(double value, double[] gradient) { this.value = value; this.gradient = gradient; }
		static Diff constant(double value, int dimension) { return new Diff(value, new double[dimension]); }
		Diff add(Diff other) { return combine(value + other.value, 1.0, other, 1.0); }
		Diff add(double other) { return new Diff(value + other, gradient.clone()); }
		Diff subtract(Diff other) { return combine(value - other.value, 1.0, other, -1.0); }
		Diff multiply(Diff other) { return combine(value * other.value, other.value, other, value); }
		Diff multiply(double other) { return scale(value * other, other); }
		Diff divide(Diff other) { return combine(value / other.value, 1.0 / other.value,
				other, -value / (other.value * other.value)); }
		Diff negate() { return scale(-value, -1.0); }
		Diff inverse() { return scale(1.0 / value, -1.0 / (value * value)); }
		Diff exp() { double result = Math.exp(value); return scale(result, result); }
		Diff log() { return scale(Math.log(value), 1.0 / value); }
		Diff sqrt() { double result = Math.sqrt(value); return scale(result, 0.5 / result); }
		Diff abs() { return scale(Math.abs(value), Math.signum(value)); }
		Diff lgamma() { return scale(lgammafn(value), psi(value)); }
		Diff log1p() { return scale(Math.log1p(value), 1.0 / (1.0 + value)); }
		Diff expm1() { return scale(Math.expm1(value), Math.exp(value)); }
		Diff sin() { return scale(Math.sin(value), Math.cos(value)); }
		Diff cos() { return scale(Math.cos(value), -Math.sin(value)); }
		Diff tan() { double result = Math.tan(value); return scale(result, 1.0 + result * result); }
		Diff asin() { return scale(Math.asin(value), 1.0 / Math.sqrt(1.0 - value * value)); }
		Diff acos() { return scale(Math.acos(value), -1.0 / Math.sqrt(1.0 - value * value)); }
		Diff atan() { return scale(Math.atan(value), 1.0 / (1.0 + value * value)); }
		Diff sinh() { return scale(Math.sinh(value), Math.cosh(value)); }
		Diff cosh() { return scale(Math.cosh(value), Math.sinh(value)); }
		Diff tanh() { double result = Math.tanh(value); return scale(result, 1.0 - result * result); }
		Diff erf() { return scale(jdistlib.math.MathFunctions.erf__(value),
				2.0 / Math.sqrt(Math.PI) * Math.exp(-value * value)); }
		Diff erfc() { return scale(jdistlib.math.MathFunctions.erfc1(0, value),
				-2.0 / Math.sqrt(Math.PI) * Math.exp(-value * value)); }
		Diff gamma() { double result = jdistlib.math.MathFunctions.gammafn(value);
			return scale(result, result * psi(value)); }
		Diff digamma() { return scale(psi(value), jdistlib.math.PolyGamma.trigamma(value)); }
		Diff trigamma() { return scale(jdistlib.math.PolyGamma.trigamma(value), tetragamma(value)); }
		Diff pow(Diff other) {
			if (other.constantGradient() && other.value == Math.rint(other.value)) {
				double result = Math.pow(value, other.value);
				return scale(result, other.value * Math.pow(value, other.value - 1.0));
			}
			return log().multiply(other).exp();
		}
		private boolean constantGradient() {
			for (double derivative : gradient) if (derivative != 0.0) return false;
			return true;
		}
		private Diff scale(double result, double multiplier) {
			double[] derivative = new double[gradient.length];
			for (int i = 0; i < derivative.length; i++) derivative[i] = multiplier * gradient[i];
			return new Diff(result, derivative);
		}
		private Diff combine(double result, double ownMultiplier, Diff other, double otherMultiplier) {
			double[] derivative = new double[gradient.length];
			for (int i = 0; i < derivative.length; i++)
				derivative[i] = ownMultiplier * gradient[i] + otherMultiplier * other.gradient[i];
			return new Diff(result, derivative);
		}
	}

	private static Diff logProbability(String distribution, Diff[] x) {
		if (distribution.equals("std_normal") && x.length == 1)
			return x[0].multiply(x[0]).multiply(-0.5).add(-0.5 * Math.log(2.0 * Math.PI));
		if (distribution.equals("normal") && x.length == 3) {
			if (!positive(x[2])) return outside(x);
			Diff z = x[0].subtract(x[1]).divide(x[2]);
			return x[2].log().negate().add(-0.5 * Math.log(2.0 * Math.PI))
					.subtract(z.multiply(z).multiply(0.5));
		}
		if (distribution.equals("lognormal") && x.length == 3) {
			if (!positive(x[0]) || !positive(x[2])) return outside(x);
			Diff logged = x[0].log(); Diff z = logged.subtract(x[1]).divide(x[2]);
			return x[2].log().negate().subtract(logged).add(-0.5 * Math.log(2.0 * Math.PI))
					.subtract(z.multiply(z).multiply(0.5));
		}
		if (distribution.equals("student_t") && x.length == 4) {
			if (!positive(x[1]) || !positive(x[3])) return outside(x);
			Diff z = x[0].subtract(x[2]).divide(x[3]);
			return x[1].add(1.0).multiply(0.5).lgamma().subtract(x[1].multiply(0.5).lgamma())
					.subtract(x[1].multiply(Math.PI).log().multiply(0.5)).subtract(x[3].log())
					.subtract(x[1].add(1.0).multiply(0.5)
							.multiply(z.multiply(z).divide(x[1]).log1p()));
		}
		if (distribution.equals("cauchy") && x.length == 3) {
			if (!positive(x[2])) return outside(x);
			Diff z = x[0].subtract(x[1]).divide(x[2]);
			return x[2].log().negate().add(-Math.log(Math.PI))
					.subtract(Diff.constant(1.0, x[0].gradient.length).add(z.multiply(z)).log());
		}
		if (distribution.equals("double_exponential") && x.length == 3) {
			if (!positive(x[2])) return outside(x);
			return x[2].log().negate().add(-Math.log(2.0))
					.subtract(x[0].subtract(x[1]).abs().divide(x[2]));
		}
		if (distribution.equals("logistic") && x.length == 3) {
			if (!positive(x[2])) return outside(x);
			Diff z = x[0].subtract(x[1]).divide(x[2]);
			return x[2].log().negate().subtract(z).subtract(log1pExp(z.negate()).multiply(2.0));
		}
		if (distribution.equals("gumbel") && x.length == 3) {
			if (!positive(x[2])) return outside(x);
			Diff z = x[0].subtract(x[1]).divide(x[2]);
			return x[2].log().negate().subtract(z).subtract(z.negate().exp());
		}
		if (distribution.equals("skew_normal") && x.length == 4) {
			if (!positive(x[2])) return outside(x);
			Diff z = x[0].subtract(x[1]).divide(x[2]);
			return x[2].log().negate().add(0.5 * Math.log(2.0 / Math.PI))
					.subtract(z.multiply(z).multiply(0.5)).add(logPhi(z.multiply(x[3])));
		}
		if (distribution.equals("exp_mod_normal") && x.length == 4) {
			if (!positive(x[2]) || !positive(x[3])) return outside(x);
			Diff difference = x[0].subtract(x[1]);
			Diff scaledRate = x[3].multiply(x[2]);
			Diff normalArgument = difference.divide(x[2]).subtract(scaledRate);
			return x[3].log().subtract(x[3].multiply(difference))
					.add(scaledRate.multiply(scaledRate).multiply(0.5))
					.add(logPhi(normalArgument));
		}
		if (distribution.equals("von_mises") && x.length == 3) {
			if (x[2].value < 0.0) return outside(x);
			return x[2].multiply(x[0].subtract(x[1]).cos()).add(-Math.log(2.0 * Math.PI))
					.subtract(logBesselI0(x[2]));
		}
		if (distribution.equals("exponential") && x.length == 2) {
			if (x[0].value < 0.0 || !positive(x[1])) return outside(x);
			return x[1].log().subtract(x[1].multiply(x[0]));
		}
		if (distribution.equals("gamma") && x.length == 3) { // Stan shape/rate
			if (!positive(x[0]) || !positive(x[1]) || !positive(x[2])) return outside(x);
			return x[1].multiply(x[2].log()).subtract(x[1].lgamma())
					.add(x[1].subtract(one(x)).multiply(x[0].log())).subtract(x[2].multiply(x[0]));
		}
		if (distribution.equals("inv_gamma") && x.length == 3) {
			if (!positive(x[0]) || !positive(x[1]) || !positive(x[2])) return outside(x);
			return x[1].multiply(x[2].log()).subtract(x[1].lgamma())
					.subtract(x[1].add(1.0).multiply(x[0].log())).subtract(x[2].divide(x[0]));
		}
		if (distribution.equals("chi_square") && x.length == 2) {
			if (!positive(x[0]) || !positive(x[1])) return outside(x);
			Diff shape = x[1].multiply(0.5); Diff rate = Diff.constant(0.5, x[0].gradient.length);
			return shape.multiply(rate.log()).subtract(shape.lgamma())
					.add(shape.subtract(one(x)).multiply(x[0].log())).subtract(rate.multiply(x[0]));
		}
		if (distribution.equals("inv_chi_square") && x.length == 2) {
			Diff[] expanded = {x[0], x[1].multiply(0.5), Diff.constant(0.5, x[0].gradient.length)};
			return logProbability("inv_gamma", expanded);
		}
		if (distribution.equals("scaled_inv_chi_square") && x.length == 3) {
			Diff[] expanded = {x[0], x[1].multiply(0.5), x[1].multiply(x[2]).multiply(x[2]).multiply(0.5)};
			return logProbability("inv_gamma", expanded);
		}
		if (distribution.equals("weibull") && x.length == 3) {
			if (!positive(x[0]) || !positive(x[1]) || !positive(x[2])) return outside(x);
			Diff ratio = x[0].divide(x[2]);
			return x[1].log().subtract(x[2].log())
					.add(x[1].subtract(one(x)).multiply(ratio.log())).subtract(ratio.pow(x[1]));
		}
		if (distribution.equals("frechet") && x.length == 3) {
			if (!positive(x[0]) || !positive(x[1]) || !positive(x[2])) return outside(x);
			return x[1].log().add(x[1].multiply(x[2].log()))
					.subtract(x[1].add(1.0).multiply(x[0].log()))
					.subtract(x[2].divide(x[0]).pow(x[1]));
		}
		if (distribution.equals("rayleigh") && x.length == 2) {
			if (x[0].value < 0.0 || !positive(x[1])) return outside(x);
			return x[0].log().subtract(x[1].log().multiply(2.0))
					.subtract(x[0].multiply(x[0]).divide(x[1].multiply(x[1]).multiply(2.0)));
		}
		if (distribution.equals("beta") && x.length == 3) {
			if (x[0].value < 0.0 || x[0].value > 1.0 || !positive(x[1]) || !positive(x[2])) return outside(x);
			return x[1].subtract(one(x)).multiply(x[0].log())
					.add(x[2].subtract(one(x)).multiply(one(x).subtract(x[0]).log()))
					.add(x[1].add(x[2]).lgamma()).subtract(x[1].lgamma()).subtract(x[2].lgamma());
		}
		if (distribution.equals("beta_proportion") && x.length == 3) {
			Diff[] expanded = {x[0], x[1].multiply(x[2]), one(x).subtract(x[1]).multiply(x[2])};
			return logProbability("beta", expanded);
		}
		if (distribution.equals("uniform") && x.length == 3) {
			if (!(x[2].value > x[1].value) || x[0].value < x[1].value || x[0].value > x[2].value)
				return outside(x);
			return x[2].subtract(x[1]).log().negate();
		}
		if (distribution.equals("pareto") && x.length == 3) {
			if (!positive(x[1]) || !positive(x[2]) || x[0].value < x[1].value) return outside(x);
			return x[2].log().add(x[2].multiply(x[1].log()))
					.subtract(x[2].add(1.0).multiply(x[0].log()));
		}
		if (distribution.equals("pareto_type_2") && x.length == 4) {
			if (!positive(x[2]) || !positive(x[3]) || x[0].value < x[1].value) return outside(x);
			return x[3].log().subtract(x[2].log()).subtract(x[3].add(1.0)
					.multiply(x[0].subtract(x[1]).divide(x[2]).log1p()));
		}
		if (distribution.equals("bernoulli") && x.length == 2) {
			if (!integerIn(x[0], 0, 1) || !probability(x[1])) return outside(x);
			return x[0].value == 1.0 ? x[1].log() : one(x).subtract(x[1]).log();
		}
		if (distribution.equals("bernoulli_logit") && x.length == 2) {
			if (!integerIn(x[0], 0, 1)) return outside(x);
			return x[0].multiply(logInvLogit(x[1])).add(one(x).subtract(x[0]).multiply(log1mInvLogit(x[1])));
		}
		if (distribution.equals("binomial") && x.length == 3) {
			if (!integerIn(x[0], 0, x[1].value) || !nonnegativeInteger(x[1]) || !probability(x[2])) return outside(x);
			Diff result = logChoose(x[1], x[0]);
			if (x[0].value > 0.0) result = result.add(x[0].multiply(x[2].log()));
			if (x[1].value > x[0].value)
				result = result.add(x[1].subtract(x[0]).multiply(one(x).subtract(x[2]).log()));
			return result;
		}
		if (distribution.equals("binomial_logit") && x.length == 3) {
			if (!integerIn(x[0], 0, x[1].value) || !nonnegativeInteger(x[1])) return outside(x);
			return logChoose(x[1], x[0]).add(x[0].multiply(x[2]))
					.subtract(x[1].multiply(log1pExp(x[2])));
		}
		if (distribution.equals("beta_binomial") && x.length == 4) {
			if (!integerIn(x[0], 0, x[1].value) || !nonnegativeInteger(x[1])
					|| !positive(x[2]) || !positive(x[3])) return outside(x);
			return logChoose(x[1], x[0]).add(logBeta(x[0].add(x[2]), x[1].subtract(x[0]).add(x[3])))
					.subtract(logBeta(x[2], x[3]));
		}
		if (distribution.equals("hypergeometric") && x.length == 4) {
			if (!nonnegativeInteger(x[0]) || !nonnegativeInteger(x[1])
					|| !nonnegativeInteger(x[2]) || !nonnegativeInteger(x[3])
					|| x[0].value > x[2].value || x[1].value - x[0].value > x[3].value)
				return outside(x);
			return logChoose(x[2], x[0]).add(logChoose(x[3], x[1].subtract(x[0])))
					.subtract(logChoose(x[2].add(x[3]), x[1]));
		}
		if (distribution.equals("neg_binomial") && x.length == 3) {
			if (!nonnegativeInteger(x[0]) || !positive(x[1]) || !positive(x[2])) return outside(x);
			return x[0].add(x[1]).lgamma().subtract(x[1].lgamma()).subtract(x[0].add(1.0).lgamma())
					.add(x[1].multiply(x[2].log())).subtract(x[0].add(x[1]).multiply(x[2].add(1.0).log()));
		}
		if (distribution.equals("neg_binomial_2") && x.length == 3) {
			if (!nonnegativeInteger(x[0]) || !positive(x[1]) || !positive(x[2])) return outside(x);
			Diff total = x[1].add(x[2]);
			return x[0].add(x[2]).lgamma().subtract(x[2].lgamma()).subtract(x[0].add(1.0).lgamma())
					.add(x[2].multiply(x[2].divide(total).log()))
					.add(x[0].multiply(x[1].divide(total).log()));
		}
		if (distribution.equals("neg_binomial_2_log") && x.length == 3) {
			Diff[] expanded = {x[0], x[1].exp(), x[2]}; return logProbability("neg_binomial_2", expanded);
		}
		if (distribution.equals("poisson") && x.length == 2) {
			if (!nonnegativeInteger(x[0]) || x[1].value < 0.0) return outside(x);
			if (x[1].value == 0.0)
				return x[0].value == 0.0 ? Diff.constant(0.0, x[0].gradient.length) : outside(x);
			return x[0].multiply(x[1].log()).subtract(x[1]).subtract(x[0].add(1.0).lgamma());
		}
		if (distribution.equals("poisson_log") && x.length == 2) {
			if (!nonnegativeInteger(x[0])) return outside(x);
			return x[0].multiply(x[1]).subtract(x[1].exp()).subtract(x[0].add(1.0).lgamma());
		}
		if (distribution.equals("geometric") && x.length == 2) {
			if (!nonnegativeInteger(x[0]) || !probability(x[1])) return outside(x);
			Diff result = x[1].log();
			return x[0].value == 0.0 ? result
					: result.add(x[0].multiply(one(x).subtract(x[1]).log()));
		}
		if (distribution.equals("discrete_range") && x.length == 3) {
			if (!integer(x[0]) || !integer(x[1]) || !integer(x[2])
					|| x[2].value < x[1].value || x[0].value < x[1].value || x[0].value > x[2].value)
				return outside(x);
			return x[2].subtract(x[1]).add(1.0).log().negate();
		}
		throw new IllegalArgumentException("unsupported distribution or arity: " + distribution);
	}

	private static Diff random(String distribution, Diff[] x, Context context) {
		if (context.random == null) throw new IllegalArgumentException(distribution + "_rng is only valid in generated quantities");
		double value;
		if (distribution.equals("std_normal") && x.length == 0)
			value = Normal.random(0.0, 1.0, context.random);
		else if (distribution.equals("normal") && x.length == 2)
			value = Normal.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("lognormal") && x.length == 2)
			value = LogNormal.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("student_t") && x.length == 3)
			value = x[1].value + x[2].value * T.random(x[0].value, context.random);
		else if (distribution.equals("double_exponential") && x.length == 2)
			value = Laplace.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("logistic") && x.length == 2)
			value = Logistic.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("gumbel") && x.length == 2)
			value = x[0].value - x[1].value * Math.log(-Math.log(context.random.nextDouble()));
		else if (distribution.equals("skew_normal") && x.length == 3) {
			double delta = x[2].value / Math.sqrt(1.0 + x[2].value * x[2].value);
			double first = Normal.random(0.0, 1.0, context.random);
			double second = Normal.random(0.0, 1.0, context.random);
			value = x[0].value + x[1].value * (delta * Math.abs(first)
					+ Math.sqrt(1.0 - delta * delta) * second);
		}
		else if (distribution.equals("exp_mod_normal") && x.length == 3)
			value = ExponentiallyModifiedGaussian.random(
					x[0].value, x[1].value, x[2].value, context.random);
		else if (distribution.equals("von_mises") && x.length == 2)
			value = vonMisesRandom(x[0].value, x[1].value, context.random);
		else if (distribution.equals("gamma") && x.length == 2)
			value = Gamma.random(x[0].value, 1.0 / x[1].value, context.random);
		else if (distribution.equals("inv_gamma") && x.length == 2)
			value = 1.0 / Gamma.random(x[0].value, 1.0 / x[1].value, context.random);
		else if (distribution.equals("chi_square") && x.length == 1)
			value = ChiSquare.random(x[0].value, context.random);
		else if (distribution.equals("inv_chi_square") && x.length == 1)
			value = 1.0 / ChiSquare.random(x[0].value, context.random);
		else if (distribution.equals("scaled_inv_chi_square") && x.length == 2)
			value = x[0].value * x[1].value * x[1].value / ChiSquare.random(x[0].value, context.random);
		else if (distribution.equals("exponential") && x.length == 1)
			value = Exponential.random(1.0 / x[0].value, context.random);
		else if (distribution.equals("weibull") && x.length == 2)
			value = Weibull.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("frechet") && x.length == 2)
			value = x[1].value / Math.pow(-Math.log(context.random.nextDouble()), 1.0 / x[0].value);
		else if (distribution.equals("rayleigh") && x.length == 1)
			value = x[0].value * Math.sqrt(-2.0 * Math.log(context.random.nextDouble()));
		else if (distribution.equals("beta") && x.length == 2)
			value = Beta.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("beta_proportion") && x.length == 2)
			value = Beta.random(x[0].value * x[1].value,
					(1.0 - x[0].value) * x[1].value, context.random);
		else if (distribution.equals("bernoulli") && x.length == 1)
			value = context.random.nextDouble() < x[0].value ? 1.0 : 0.0;
		else if (distribution.equals("bernoulli_logit") && x.length == 1)
			value = context.random.nextDouble() < invLogit(x[0].value) ? 1.0 : 0.0;
		else if (distribution.equals("binomial") && x.length == 2)
			value = Binomial.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("binomial_logit") && x.length == 2)
			value = Binomial.random(x[0].value, invLogit(x[1].value), context.random);
		else if (distribution.equals("beta_binomial") && x.length == 3)
			value = Binomial.random(x[0].value,
					Beta.random(x[1].value, x[2].value, context.random), context.random);
		else if (distribution.equals("hypergeometric") && x.length == 3)
			value = HyperGeometric.random(x[1].value, x[2].value, x[0].value, context.random);
		else if (distribution.equals("neg_binomial") && x.length == 2)
			value = NegBinomial.random(x[0].value, x[1].value / (1.0 + x[1].value), context.random);
		else if (distribution.equals("neg_binomial_2") && x.length == 2)
			value = NegBinomial.random(x[1].value, x[1].value / (x[0].value + x[1].value), context.random);
		else if (distribution.equals("neg_binomial_2_log") && x.length == 2)
			value = NegBinomial.random(x[1].value,
					x[1].value / (Math.exp(x[0].value) + x[1].value), context.random);
		else if (distribution.equals("poisson") && x.length == 1)
			value = Poisson.random(x[0].value, context.random);
		else if (distribution.equals("poisson_log") && x.length == 1)
			value = Poisson.random(Math.exp(x[0].value), context.random);
		else if (distribution.equals("geometric") && x.length == 1)
			value = Geometric.random(x[0].value, context.random);
		else if (distribution.equals("uniform") && x.length == 2)
			value = Uniform.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("pareto") && x.length == 2)
			value = x[0].value / Math.pow(1.0 - context.random.nextDouble(), 1.0 / x[1].value);
		else if (distribution.equals("pareto_type_2") && x.length == 3)
			value = x[0].value + x[1].value
					* (Math.pow(1.0 - context.random.nextDouble(), -1.0 / x[2].value) - 1.0);
		else if (distribution.equals("discrete_range") && x.length == 2)
			value = x[0].value + Math.floor(context.random.nextDouble() * (x[1].value - x[0].value + 1.0));
		else if (distribution.equals("cauchy") && x.length == 2)
			value = Cauchy.random(x[0].value, x[1].value, context.random);
		else throw new IllegalArgumentException("unsupported RNG or arity: " + distribution);
		return Diff.constant(value, context.dimension);
	}

	private static boolean positive(Diff value) { return value.value > 0.0; }
	private static boolean probability(Diff value) {
		return value.value >= 0.0 && value.value <= 1.0;
	}
	private static boolean integer(Diff value) { return value.value == Math.rint(value.value); }
	private static boolean nonnegativeInteger(Diff value) {
		return integer(value) && value.value >= 0.0;
	}
	private static boolean integerIn(Diff value, double lower, double upper) {
		return integer(value) && value.value >= lower && value.value <= upper;
	}
	private static Diff outside(Diff[] values) {
		int dimension = values.length == 0 ? 0 : values[0].gradient.length;
		return Diff.constant(Double.NEGATIVE_INFINITY, dimension);
	}
	private static Diff one(Diff[] values) {
		return Diff.constant(1.0, values[0].gradient.length);
	}
	private static Diff logChoose(Diff total, Diff selected) {
		return total.add(1.0).lgamma().subtract(selected.add(1.0).lgamma())
				.subtract(total.subtract(selected).add(1.0).lgamma());
	}
	private static Diff logBeta(Diff first, Diff second) {
		return first.lgamma().add(second.lgamma()).subtract(first.add(second).lgamma());
	}
	private static Diff log1pExp(Diff value) {
		double result = log1pExp(value.value);
		return value.scale(result, invLogit(value.value));
	}
	private static Diff logInvLogit(Diff value) {
		return value.scale(-log1pExp(-value.value), invLogit(-value.value));
	}
	private static Diff scalarInvLogit(Diff value) {
		double result = invLogit(value.value);
		return value.scale(result, result * (1.0 - result));
	}
	private static Diff log1mInvLogit(Diff value) {
		return value.scale(-log1pExp(value.value), -invLogit(value.value));
	}
	private static Diff phi(Diff value) {
		return value.scale(Normal.cumulative(value.value, 0.0, 1.0, true, false),
				Normal.density(value.value, 0.0, 1.0, false));
	}
	private static Diff logPhi(Diff value) {
		double result = Normal.cumulative(value.value, 0.0, 1.0, true, true);
		double logDensity = Normal.density(value.value, 0.0, 1.0, true);
		return value.scale(result, Math.exp(logDensity - result));
	}
	private static Diff logBesselI0(Diff value) {
		double scaledZero = jdistlib.math.Bessel.i(value.value, 0.0, true);
		double scaledOne = jdistlib.math.Bessel.i(value.value, 1.0, true);
		return value.scale(Math.log(scaledZero) + value.value, scaledOne / scaledZero);
	}
	private static double vonMisesRandom(double location, double concentration,
			RandomEngine random) {
		if (!(concentration >= 0.0)) return Double.NaN;
		if (concentration < 1e-8)
			return wrapAngle(location + (2.0 * random.nextDouble() - 1.0) * Math.PI);
		double a = 1.0 + Math.sqrt(1.0 + 4.0 * concentration * concentration);
		double b = (a - Math.sqrt(2.0 * a)) / (2.0 * concentration);
		double r = (1.0 + b * b) / (2.0 * b);
		for (;;) {
			double z = Math.cos(Math.PI * random.nextDouble());
			double f = (1.0 + r * z) / (r + z);
			double c = concentration * (r - f);
			double second = random.nextDouble();
			if (second < c * (2.0 - c) || Math.log(c / second) + 1.0 - c >= 0.0) {
				double angle = random.nextDouble() > 0.5 ? Math.acos(f) : -Math.acos(f);
				return wrapAngle(location + angle);
			}
		}
	}
	private static double wrapAngle(double angle) {
		double wrapped = angle % (2.0 * Math.PI);
		if (wrapped > Math.PI) wrapped -= 2.0 * Math.PI;
		if (wrapped < -Math.PI) wrapped += 2.0 * Math.PI;
		return wrapped;
	}

	private static Diff unary(Diff[] values, String name) {
		if (values.length != 1) throw new IllegalArgumentException(name + " expects one argument");
		return values[0];
	}
	private static Diff[] binary(Diff[] values, String name) {
		if (values.length != 2) throw new IllegalArgumentException(name + " expects two arguments");
		return values;
	}
	private static Diff[] ternary(Diff[] values, String name) {
		if (values.length != 3) throw new IllegalArgumentException(name + " expects three arguments");
		return values;
	}
	private static Diff constantFunction(Diff[] values, String name, double value, Context context) {
		if (values.length != 0) throw new IllegalArgumentException(name + " expects no arguments");
		return Diff.constant(value, context.dimension);
	}
	private static Diff nondifferentiable(Diff[] values, String name, double value) {
		Diff argument = unary(values, name);
		return Diff.constant(value, argument.gradient.length);
	}
	private static Diff predicate(Diff[] values, String name, boolean value, Context context) {
		unary(values, name);
		return booleanValue(value, context);
	}
	private static Diff booleanValue(boolean value, Context context) {
		return Diff.constant(value ? 1.0 : 0.0, context.dimension);
	}
	private static boolean truth(Diff value) {
		return value.value != 0.0 && !Double.isNaN(value.value);
	}
	private static double invLogit(double value) {
		if (value >= 0.0) { double exponential = Math.exp(-value); return 1.0 / (1.0 + exponential); }
		double exponential = Math.exp(value); return exponential / (1.0 + exponential);
	}
	private static double log1pExp(double value) {
		return value > 0.0 ? value + Math.log1p(Math.exp(-value)) : Math.log1p(Math.exp(value));
	}
	private static double log1mExp(double value) {
		if (value > 0.0) return Double.NaN;
		return value < -Math.log(2.0) ? Math.log1p(-Math.exp(value)) : Math.log(-Math.expm1(value));
	}

	private static final class Parser {
		final Lexer lexer; Token current; final List<ScriptDiagnostic> diagnostics = new ArrayList<ScriptDiagnostic>();
		Parser(String source) { lexer = new Lexer(source); current = lexer.next(); }
		Program parse() {
			Program program = new Program();
			while (current.kind != TokenKind.EOF) {
				try { parseBlock(program); }
				catch (ParseFailure failure) { diagnostics.add(failure.diagnostic); synchronize(); }
			}
			return program;
		}
		private void parseBlock(Program program) {
			Token start = require(TokenKind.IDENTIFIER, "block name expected");
			String block = start.text;
			if (block.equals("transformed") || block.equals("generated")) {
				Token second = require(TokenKind.IDENTIFIER, "second block-name word expected");
				block += " " + second.text;
			}
			require("{");
			while (!accept("}")) {
				if (current.kind == TokenKind.EOF) fail(current, "unterminated " + block + " block");
				if (block.equals("data")) program.data.add(parseDeclaration());
				else if (block.equals("parameters")) program.parameters.add(parseDeclaration());
				else if (block.equals("transformed data")) program.transformedData.add(parseAssignment(true));
				else if (block.equals("transformed parameters")) program.transformedParameters.add(parseAssignment(true));
				else if (block.equals("generated quantities")) program.generated.add(parseAssignment(true));
				else if (block.equals("model")) program.model.add(parseStatement());
				else fail(start, "unknown block '" + block + "'");
			}
		}
		private Declaration parseDeclaration() {
			Token typeToken = require(TokenKind.IDENTIFIER, "declaration type expected");
			String type = typeToken.text;
			if (!(type.equals("real") || type.equals("int") || type.equals("vector")
					|| type.equals("simplex") || type.equals("ordered"))) fail(typeToken, "unsupported declaration type");
			Expr dimension = null;
			if (accept("[")) { dimension = expression(0); require("]"); }
			Expr lower = null, upper = null;
			if (accept("<")) {
				do {
					Token bound = require(TokenKind.IDENTIFIER, "constraint name expected"); require("=");
					Expr value = expression(4);
					if (bound.text.equals("lower")) lower = value;
					else if (bound.text.equals("upper")) upper = value;
					else fail(bound, "only lower and upper constraints are supported");
				} while (accept(","));
				require(">");
			}
			Token name = require(TokenKind.IDENTIFIER, "declaration name expected");
			if (dimension == null && accept("[")) { dimension = expression(0); require("]"); }
			require(";");
			return new Declaration(type, name.text, dimension, lower, upper,
					type.equals("int"), name);
		}
		private Assignment parseAssignment(boolean declarationAllowed) {
			if (declarationAllowed && current.kind == TokenKind.IDENTIFIER
					&& (current.text.equals("real") || current.text.equals("int"))) advance();
			Token name = require(TokenKind.IDENTIFIER, "assignment name expected");
			require("="); Expr value = expression(0); require(";");
			return new Assignment(name.text, value);
		}
		private Statement parseStatement() {
			if (accept("{")) {
				List<Statement> statements = new ArrayList<Statement>();
				while (!accept("}")) {
					if (current.kind == TokenKind.EOF) fail(current, "unterminated statement block");
					statements.add(parseStatement());
				}
				return new BlockStatement(statements, true);
			}
			if (current.text.equals("if")) {
				advance(); require("("); Expr condition = expression(0); require(")");
				Statement whenTrue = parseStatement();
				Statement whenFalse = null;
				if (current.text.equals("else")) { advance(); whenFalse = parseStatement(); }
				return new IfStatement(condition, whenTrue, whenFalse);
			}
			if (current.text.equals("for")) {
				advance(); require("(");
				Token variable = require(TokenKind.IDENTIFIER, "loop variable expected");
				if (!current.text.equals("in")) fail(current, "expected 'in'");
				advance(); Expr lower = expression(0); require(":"); Expr upper = expression(0); require(")");
				return new ForStatement(variable.text, lower, upper, parseStatement());
			}
			if (current.text.equals("while")) {
				advance(); require("("); Expr condition = expression(0); require(")");
				return new WhileStatement(condition, parseStatement());
			}
			if (current.text.equals("real") || current.text.equals("int")) {
				boolean integer = current.text.equals("int"); advance();
				Token name = require(TokenKind.IDENTIFIER, "local variable name expected");
				require("="); Expr initializer = expression(0); require(";");
				return new LocalDeclarationStatement(name.text, initializer, integer);
			}
			if (current.text.equals("target")) {
				advance(); require("+="); Expr value = expression(0); require(";");
				return new TargetStatement(value);
			}
			Expr left = expression(0);
			if (accept("~")) {
				Token distribution = require(TokenKind.IDENTIFIER, "distribution name expected");
				require("("); List<Expr> arguments = arguments(); require(")"); require(";");
				return new SamplingStatement(left, distribution.text, arguments);
			}
			if (!(left instanceof VariableExpr) || ((VariableExpr) left).index != null)
				fail(current, "assignment requires a scalar local variable");
			String operator = current.text;
			if (!(operator.equals("=") || operator.equals("+=") || operator.equals("-=")
					|| operator.equals("*=") || operator.equals("/=")))
				fail(current, "expected sampling or assignment operator");
			advance(); Expr value = expression(0); require(";");
			return new AssignmentStatement(((VariableExpr) left).name, operator, value);
		}
		private Expr expression(int minimumPrecedence) {
			Expr left;
			if (accept("+") || accept("-") || accept("!")) {
				String operator = previous; left = new UnaryExpr(operator, expression(7));
			} else if (accept("(")) { left = expression(0); require(")"); }
			else if (current.kind == TokenKind.NUMBER) {
				Token number = current; advance();
				try { left = new NumberExpr(Double.parseDouble(number.text)); }
				catch (NumberFormatException exception) { fail(number, "invalid number"); return null; }
			} else if (current.kind == TokenKind.IDENTIFIER) {
				Token identifier = current; advance();
				if (accept("(")) { List<Expr> args = arguments(); require(")"); left = new CallExpr(identifier.text, args); }
				else {
					Expr index = null; if (accept("[")) { index = expression(0); require("]"); }
					left = new VariableExpr(identifier.text, index);
				}
			} else { fail(current, "expression expected"); return null; }
			while (true) {
				int precedence = precedence(current.text);
				if (precedence < minimumPrecedence) break;
				String operator = current.text; advance();
				left = new BinaryExpr(operator, left, expression(precedence + (operator.equals("^") ? 0 : 1)));
			}
			return left;
		}
		private List<Expr> arguments() {
			List<Expr> result = new ArrayList<Expr>();
			if (current.text.equals(")")) return result;
			result.add(expression(0));
			while (accept(",") || accept("|")) result.add(expression(0));
			return result;
		}
		private int precedence(String operator) {
			if (operator.equals("||")) return 0;
			if (operator.equals("&&")) return 1;
			if (operator.equals("==") || operator.equals("!=")) return 2;
			if (operator.equals("<") || operator.equals("<=")
					|| operator.equals(">") || operator.equals(">=")) return 3;
			if (operator.equals("+") || operator.equals("-")) return 4;
			if (operator.equals("*") || operator.equals("/")) return 5;
			if (operator.equals("^")) return 6;
			return -1;
		}
		private String previous;
		private boolean accept(String text) {
			if (!current.text.equals(text)) return false;
			previous = current.text; advance(); return true;
		}
		private void require(String text) { if (!accept(text)) fail(current, "expected '" + text + "'"); }
		private Token require(TokenKind kind, String message) {
			if (current.kind != kind) fail(current, message);
			Token result = current; advance(); return result;
		}
		private void advance() { current = lexer.next(); }
		private void fail(Token token, String message) { throw new ParseFailure(new ScriptDiagnostic(token.line, token.column, message)); }
		private void synchronize() {
			while (current.kind != TokenKind.EOF && !current.text.equals(";") && !current.text.equals("}")) advance();
			if (current.text.equals(";")) advance(); else if (current.text.equals("}")) advance();
		}
	}
	private static final class ParseFailure extends RuntimeException {
		private static final long serialVersionUID = 1L; final ScriptDiagnostic diagnostic;
		ParseFailure(ScriptDiagnostic diagnostic) { this.diagnostic = diagnostic; }
	}

	private static int constrainedDimension(List<Declaration> declarations, Context constants) {
		int result = 0;
		for (Declaration declaration : declarations) result += declaration.dimension == null ? 1
				: checkedDimension(declaration.dimension.eval(constants).value, declaration);
		return result;
	}
	private static int checkedDimension(double value, Declaration declaration) {
		if (value != Math.rint(value) || value < 1 || value > 100000)
			throw new ModelScriptException(Collections.singletonList(declaration.error("dimension must be a practical positive integer")));
		return (int) value;
	}
	private static int checkedIndex(double value, int length, String name) {
		if (value != Math.rint(value) || value < 1 || value > length)
			throw new IllegalArgumentException("index for " + name + " must be in 1.." + length);
		return (int) value - 1;
	}
	private static int checkedLoopBound(double value, String variable) {
		if (value != Math.rint(value) || value < Integer.MIN_VALUE || value > Integer.MAX_VALUE)
			throw new IllegalArgumentException("loop bound for " + variable + " must be an integer");
		return (int) value;
	}
	private static Map<String, double[]> copyData(Map<String, double[]> source) {
		Map<String, double[]> result = new LinkedHashMap<String, double[]>();
		for (Map.Entry<String, double[]> entry : source.entrySet()) {
			if (entry.getKey() == null || entry.getValue() == null)
				throw new IllegalArgumentException("data names and values must not be null");
			result.put(entry.getKey(), entry.getValue().clone());
		}
		return result;
	}
}
