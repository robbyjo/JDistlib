/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import static jdistlib.math.MathFunctions.lgammafn;
import static jdistlib.math.MathFunctions.psi;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import jdistlib.Binomial;
import jdistlib.Cauchy;
import jdistlib.Exponential;
import jdistlib.Gamma;
import jdistlib.Normal;
import jdistlib.Poisson;
import jdistlib.Uniform;
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
				if (pair.equals("+=") || pair.equals("<=") || pair.equals(">=")
						|| pair.equals("==") || pair.equals("!=")) {
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
				constants.locals.put(assignment.name, new Diff[] {value});
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
			int statementIndex = 0;
			for (final Statement statement : model) {
				final String factorName = "script:" + (++statementIndex) + ":" + statement.label();
				Set<String> dependencies = new LinkedHashSet<String>(); statement.collect(dependencies);
				boolean expanded;
				do {
					expanded = false;
					for (Assignment assignment : transformedParameters) {
						if (dependencies.remove(assignment.name)) {
							assignment.expression.collect(dependencies);
							expanded = true;
						}
					}
				} while (expanded);
				dependencies.retainAll(declaredNames);
				final Map<String, double[]> capturedData = copyData(dataValues);
				final List<Assignment> capturedTransforms = new ArrayList<Assignment>(transformedParameters);
				builder.factor(factorName, dependencies.toArray(new String[dependencies.size()]),
						new DifferentiableModelFactor() {
					@Override public double logDensityAndAddGradient(ModelState state,
							double[] gradient) {
						Context context = new Context(state, capturedData, constrainedDimension, null);
						for (Assignment assignment : capturedTransforms)
							context.locals.put(assignment.name, new Diff[] {assignment.expression.eval(context)});
						Diff contribution = statement.eval(context);
						for (int i = 0; i < gradient.length; i++) gradient[i] += contribution.gradient[i];
						return contribution.value;
					}
				});
			}
			final BayesianModel compiled = builder.build();
			final Map<String, double[]> capturedData = copyData(dataValues);
			final List<Assignment> capturedTransforms = new ArrayList<Assignment>(transformedParameters);
			final List<Assignment> capturedGenerated = new ArrayList<Assignment>(generated);
			CompiledModelScript.Generator generator = new CompiledModelScript.Generator() {
				@Override public Map<String, double[]> generate(ModelState state, RandomEngine random) {
					Context context = new Context(state, capturedData, constrainedDimension, random);
					for (Assignment assignment : capturedTransforms)
						context.locals.put(assignment.name, new Diff[] {assignment.expression.eval(context)});
					Map<String, double[]> result = new LinkedHashMap<String, double[]>();
					for (Assignment assignment : capturedGenerated) {
						Diff value = assignment.expression.eval(context);
						context.locals.put(assignment.name, new Diff[] {value});
						result.put(assignment.name, new double[] {value.value});
					}
					return result;
				}
			};
			return new CompiledModelScript(compiled, generator, LANGUAGE_VERSION);
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
	}
	private static final class TargetStatement implements Statement {
		final Expr expression;
		TargetStatement(Expr expression) { this.expression = expression; }
		@Override public Diff eval(Context context) { return expression.eval(context); }
		@Override public void collect(Set<String> names) { expression.collect(names); }
		@Override public String label() { return "target"; }
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
			Diff[] local = context.locals.get(name);
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
			Diff value = operand.eval(context); return operator.equals("-") ? value.negate() : value;
		}
		@Override public void collect(Set<String> names) { operand.collect(names); }
	}
	private static final class BinaryExpr implements Expr {
		final String operator; final Expr left; final Expr right;
		BinaryExpr(String operator, Expr left, Expr right) {
			this.operator = operator; this.left = left; this.right = right;
		}
		@Override public Diff eval(Context context) {
			Diff a = left.eval(context), b = right.eval(context);
			if (operator.equals("+")) return a.add(b);
			if (operator.equals("-")) return a.subtract(b);
			if (operator.equals("*")) return a.multiply(b);
			if (operator.equals("/")) return a.divide(b);
			if (operator.equals("^")) return a.pow(b);
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
			if (name.equals("exp")) return unary(values, name).exp();
			if (name.equals("log")) return unary(values, name).log();
			if (name.equals("sqrt")) return unary(values, name).sqrt();
			if (name.equals("abs")) return unary(values, name).abs();
			if (name.equals("lgamma")) return unary(values, name).lgamma();
			if (name.equals("inv_logit")) return unary(values, name).negate().exp().add(1.0).inverse();
			if (name.endsWith("_lpdf") || name.endsWith("_lpmf")) {
				String distribution = name.substring(0, name.length() - 5);
				return logProbability(distribution, values);
			}
			if (name.endsWith("_rng")) return random(name.substring(0, name.length() - 4), values, context);
			throw new IllegalArgumentException("unknown function: " + name);
		}
		@Override public void collect(Set<String> names) { for (Expr argument : arguments) argument.collect(names); }
	}

	private static final class Context {
		final ModelState state; final Map<String, double[]> data; final int dimension;
		final RandomEngine random; final Map<String, Diff[]> locals = new LinkedHashMap<String, Diff[]>();
		Context(ModelState state, Map<String, double[]> data, int dimension, RandomEngine random) {
			this.state = state; this.data = data; this.dimension = dimension; this.random = random;
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
		Diff pow(Diff other) { return log().multiply(other).exp(); }
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
		if (distribution.equals("normal") && x.length == 3) {
			Diff z = x[0].subtract(x[1]).divide(x[2]);
			return x[2].log().negate().add(-0.5 * Math.log(2.0 * Math.PI))
					.subtract(z.multiply(z).multiply(0.5));
		}
		if (distribution.equals("beta") && x.length == 3)
			return x[1].subtract(Diff.constant(1.0, x[0].gradient.length)).multiply(x[0].log())
					.add(x[2].subtract(Diff.constant(1.0, x[0].gradient.length)).multiply(
							Diff.constant(1.0, x[0].gradient.length).subtract(x[0]).log()))
					.add(x[1].add(x[2]).lgamma()).subtract(x[1].lgamma()).subtract(x[2].lgamma());
		if (distribution.equals("gamma") && x.length == 3) // Stan shape/rate
			return x[1].multiply(x[2].log()).subtract(x[1].lgamma())
					.add(x[1].subtract(Diff.constant(1.0, x[0].gradient.length)).multiply(x[0].log()))
					.subtract(x[2].multiply(x[0]));
		if (distribution.equals("exponential") && x.length == 2)
			return x[1].log().subtract(x[1].multiply(x[0]));
		if (distribution.equals("bernoulli") && x.length == 2)
			return x[0].multiply(x[1].log()).add(
					Diff.constant(1.0, x[0].gradient.length).subtract(x[0]).multiply(
							Diff.constant(1.0, x[0].gradient.length).subtract(x[1]).log()));
		if (distribution.equals("binomial") && x.length == 3)
			return x[1].add(1.0).lgamma().subtract(x[0].add(1.0).lgamma())
					.subtract(x[1].subtract(x[0]).add(1.0).lgamma())
					.add(x[0].multiply(x[2].log())).add(x[1].subtract(x[0]).multiply(
							Diff.constant(1.0, x[0].gradient.length).subtract(x[2]).log()));
		if (distribution.equals("poisson") && x.length == 2)
			return x[0].multiply(x[1].log()).subtract(x[1]).subtract(x[0].add(1.0).lgamma());
		if (distribution.equals("uniform") && x.length == 3)
			return x[2].subtract(x[1]).log().negate();
		if (distribution.equals("cauchy") && x.length == 3) {
			Diff z = x[0].subtract(x[1]).divide(x[2]);
			return x[2].log().negate().add(-Math.log(Math.PI))
					.subtract(Diff.constant(1.0, x[0].gradient.length).add(z.multiply(z)).log());
		}
		throw new IllegalArgumentException("unsupported distribution or arity: " + distribution);
	}

	private static Diff random(String distribution, Diff[] x, Context context) {
		if (context.random == null) throw new IllegalArgumentException(distribution + "_rng is only valid in generated quantities");
		double value;
		if (distribution.equals("normal") && x.length == 2)
			value = Normal.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("gamma") && x.length == 2)
			value = Gamma.random(x[0].value, 1.0 / x[1].value, context.random);
		else if (distribution.equals("exponential") && x.length == 1)
			value = Exponential.random(1.0 / x[0].value, context.random);
		else if (distribution.equals("bernoulli") && x.length == 1)
			value = context.random.nextDouble() < x[0].value ? 1.0 : 0.0;
		else if (distribution.equals("binomial") && x.length == 2)
			value = Binomial.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("poisson") && x.length == 1)
			value = Poisson.random(x[0].value, context.random);
		else if (distribution.equals("uniform") && x.length == 2)
			value = Uniform.random(x[0].value, x[1].value, context.random);
		else if (distribution.equals("cauchy") && x.length == 2)
			value = Cauchy.random(x[0].value, x[1].value, context.random);
		else throw new IllegalArgumentException("unsupported RNG or arity: " + distribution);
		return Diff.constant(value, context.dimension);
	}

	private static Diff unary(Diff[] values, String name) {
		if (values.length != 1) throw new IllegalArgumentException(name + " expects one argument");
		return values[0];
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
					Expr value = expression(0);
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
			if (current.text.equals("target")) {
				advance(); require("+="); Expr value = expression(0); require(";");
				return new TargetStatement(value);
			}
			Expr left = expression(0);
			require("~"); Token distribution = require(TokenKind.IDENTIFIER, "distribution name expected");
			require("("); List<Expr> arguments = arguments(); require(")"); require(";");
			return new SamplingStatement(left, distribution.text, arguments);
		}
		private Expr expression(int minimumPrecedence) {
			Expr left;
			if (accept("+") || accept("-")) {
				String operator = previous; left = new UnaryExpr(operator, expression(4));
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
			do { result.add(expression(0)); } while (accept(","));
			return result;
		}
		private int precedence(String operator) {
			if (operator.equals("+") || operator.equals("-")) return 1;
			if (operator.equals("*") || operator.equals("/")) return 2;
			if (operator.equals("^")) return 3;
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
