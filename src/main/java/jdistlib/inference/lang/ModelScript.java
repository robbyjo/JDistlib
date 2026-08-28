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
import jdistlib.inference.autodiff.ReverseTape;
import jdistlib.inference.solver.AlgebraicSolver;
import jdistlib.inference.solver.DaeSolver;
import jdistlib.inference.solver.OdeSolver;
import jdistlib.inference.solver.SensitivityResult;
import jdistlib.inference.solver.StiffOdeSolver;
import jdistlib.math.Integrate;
import jdistlib.math.IntegrationResult;
import jdistlib.rng.RandomEngine;

/** Java-native compiler for the JDistlib language and its Stan-compatible source core. */
public final class ModelScript {
	public static final String LANGUAGE_VERSION = "0.8";
	public static final String STAN_SOURCE_COMPATIBILITY = "core-2026-08-v0.8.3";
	private ModelScript() {}

	public static CompiledModelScript compile(String source, Map<String, double[]> suppliedData) {
		return compile(source, suppliedData, ExternalFunctionRegistry.empty());
	}

	/** Compiles with Java bindings for forward-declared external functions. */
	public static CompiledModelScript compile(String source, Map<String, double[]> suppliedData,
			ExternalFunctionRegistry externalFunctions) {
		if (source == null || suppliedData == null)
			throw new IllegalArgumentException("source and data are required");
		if (externalFunctions == null) throw new IllegalArgumentException("external registry is required");
		Parser parser = new Parser(source);
		Program program = parser.parse();
		if (!parser.diagnostics.isEmpty()) throw new ModelScriptException(parser.diagnostics);
		return program.compile(suppliedData, externalFunctions);
	}

	public static CompiledModelScript compile(String source) {
		return compile(source, Collections.<String, double[]>emptyMap());
	}

	/** Compiles ordinary Stan syntax supported by the Java-native compatibility core. */
	public static CompiledModelScript compileStan(String source,
			Map<String, double[]> suppliedData) {
		return compile(source, suppliedData);
	}

	/** Compiles Stan-compatible source with Java external-function bindings. */
	public static CompiledModelScript compileStan(String source, Map<String, double[]> suppliedData,
			ExternalFunctionRegistry externalFunctions) {
		return compile(source, suppliedData, externalFunctions);
	}

	/** Compiles a data-free Stan source program supported by the compatibility core. */
	public static CompiledModelScript compileStan(String source) { return compile(source); }

	/** Parses the source without requiring data values or constructing a model. */
	public static void validateSyntax(String source) {
		if (source == null) throw new IllegalArgumentException("source is required");
		Parser parser = new Parser(source);
		parser.parse();
		if (!parser.diagnostics.isEmpty()) throw new ModelScriptException(parser.diagnostics);
	}

	/** Validates Stan-compatible source without binding data or constructing a model. */
	public static void validateStanSyntax(String source) { validateSyntax(source); }

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
				boolean decimal = false;
				if (c == '.') { decimal = true; advance(); }
				while (index < source.length() && Character.isDigit(source.charAt(index))) advance();
				if (!decimal && index < source.length() && source.charAt(index) == '.') {
					decimal = true; advance();
					while (index < source.length() && Character.isDigit(source.charAt(index))) advance();
				}
				if (index < source.length() && (source.charAt(index) == 'e' || source.charAt(index) == 'E')) {
					advance();
					if (index < source.length() && (source.charAt(index) == '+' || source.charAt(index) == '-')) advance();
					while (index < source.length() && Character.isDigit(source.charAt(index))) advance();
				}
				if (index < source.length() && source.charAt(index) == 'i') advance();
				return new Token(TokenKind.NUMBER, source.substring(start, index), startLine, startColumn);
			}
			if (index + 1 < source.length()) {
				String pair = source.substring(index, index + 2);
				if (pair.equals("+=") || pair.equals("-=") || pair.equals("*=")
						|| pair.equals("/=") || pair.equals("<=") || pair.equals(">=")
						|| pair.equals("==") || pair.equals("!=")
						|| pair.equals("&&") || pair.equals("||")
						|| pair.equals(".*") || pair.equals("./")) {
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
		final Map<String, List<UserFunction>> functions =
				new LinkedHashMap<String, List<UserFunction>>();
		final List<Declaration> data = new ArrayList<Declaration>();
		final List<Declaration> parameters = new ArrayList<Declaration>();
		final List<Assignment> transformedData = new ArrayList<Assignment>();
		final List<Assignment> transformedParameters = new ArrayList<Assignment>();
		final List<Statement> model = new ArrayList<Statement>();
		final List<Assignment> generated = new ArrayList<Assignment>();

		CompiledModelScript compile(Map<String, double[]> supplied, ExternalFunctionRegistry externals) {
			for (List<UserFunction> overloads : functions.values()) for (UserFunction function : overloads)
				if (function.body == null && !externals.contains(function.name))
					throw new ModelScriptException(Collections.singletonList(
						new ScriptDiagnostic(function.token.line, function.token.column,
								"forward-declared function '" + function.name + "' has no definition")));
			for (List<UserFunction> overloads : functions.values()) for (UserFunction function : overloads)
				if (function.body == null) function.external = externals.function(function.name);
			Map<String, double[]> dataValues = copyData(supplied);
			Map<String, int[]> shapes = defaultShapes(dataValues);
			Map<String, ValueType> types = defaultTypes(dataValues);
			for (Declaration declaration : data) types.put(declaration.name, declaration.valueType());
			for (Declaration declaration : parameters) types.put(declaration.name, declaration.valueType());
			List<ScriptDiagnostic> diagnostics = new ArrayList<ScriptDiagnostic>();
			Context validationContext = new Context(null, dataValues, shapes, types, 0, null, functions);
			for (Declaration declaration : data) {
				double[] values = dataValues.get(declaration.name);
				if (values == null) diagnostics.add(declaration.error("missing required data '" + declaration.name + "'"));
				else {
					int[] declaredShape = checkedShape(declaration, validationContext);
					int expected = elementCount(declaredShape, declaration) * storageWidth(declaration.valueType().kind);
					shapes.put(declaration.name, declaredShape);
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
			Context constants = new Context(null, dataValues, shapes, types, 0, null, functions);
			for (Assignment assignment : transformedData) {
				RuntimeValue runtime = assignment.runtimeValue(constants); Diff[] value = runtime.values;
				double[] plain = values(value); dataValues.put(assignment.name, plain);
				builder.data(assignment.name, plain);
				int[] assignmentShape = runtime.shape;
				shapes.put(assignment.name, assignmentShape); types.put(assignment.name, assignment.valueType());
				constants.setLocal(assignment.name, runtime, true);
			}
			for (Declaration declaration : parameters) {
				int[] declaredShape = checkedShape(declaration, constants);
				int logicalDimension = elementCount(declaredShape, declaration);
				int dimension = logicalDimension * storageWidth(declaration.valueType().kind);
				shapes.put(declaration.name, declaredShape);
				ParameterConstraint constraint;
				double[] initial;
				if (complexKind(declaration.valueType().kind)) {
					if (declaration.lower != null || declaration.upper != null || declaration.offset != null
							|| declaration.multiplier != null)
						throw new ModelScriptException(Collections.singletonList(
								declaration.error("complex parameters do not accept real constraints")));
					constraint = dimension == 1 ? Constraints.real() : Constraints.realVector(dimension);
					initial = new double[dimension];
				} else if (declaration.type.equals("unit_vector")) {
					int size = declaration.baseDimension(constants, 0);
					constraint = Constraints.unitVector(size); initial = new double[size]; initial[0] = 1.0;
				} else if (declaration.type.equals("cov_matrix")) {
					int size = declaration.baseDimension(constants, 0);
					constraint = Constraints.covarianceMatrix(size); initial = identityMatrix(size);
				} else if (declaration.type.equals("corr_matrix")) {
					int size = declaration.baseDimension(constants, 0);
					constraint = Constraints.correlationMatrix(size); initial = identityMatrix(size);
				} else if (declaration.type.equals("cholesky_factor_corr")) {
					int size = declaration.baseDimension(constants, 0);
					constraint = Constraints.choleskyFactorCorrelation(size); initial = identityMatrix(size);
				} else if (declaration.type.equals("cholesky_factor_cov")) {
					int rows = declaration.baseDimension(constants, 0);
					int columns = declaration.baseDimension(constants, 1);
					constraint = Constraints.choleskyFactorCovariance(rows, columns);
					initial = new double[rows * columns];
					for (int i = 0; i < columns; i++) initial[i * columns + i] = 1.0;
				} else if (declaration.type.equals("simplex")) {
					int size = declaration.baseDimension(constants, 0);
					constraint = Constraints.simplex(size); initial = new double[size];
					Arrays.fill(initial, 1.0 / size);
				} else if (declaration.type.equals("ordered")) {
					int size = declaration.baseDimension(constants, 0);
					constraint = Constraints.ordered(size); initial = new double[size];
					for (int i = 0; i < size; i++) initial[i] = i;
				} else if (declaration.type.equals("positive_ordered")) {
					int size = declaration.baseDimension(constants, 0);
					constraint = Constraints.positiveOrdered(size); initial = new double[size];
					for (int i = 0; i < size; i++) initial[i] = i + 1.0;
				} else if (declaration.type.equals("sum_to_zero_vector")) {
					int size = declaration.baseDimension(constants, 0);
					constraint = Constraints.sumToZero(size); initial = new double[size];
				} else if (declaration.offset != null || declaration.multiplier != null) {
					double offset = declaration.offset == null ? 0.0 : declaration.offset.eval(constants).value;
					double multiplier = declaration.multiplier == null ? 1.0
							: declaration.multiplier.eval(constants).value;
					constraint = Constraints.offsetMultiplier(offset, multiplier, dimension);
					initial = new double[dimension]; Arrays.fill(initial, offset);
				} else if (declaration.lower != null && declaration.upper != null) {
					double lower = declaration.lower.eval(constants).value;
					double upper = declaration.upper.eval(constants).value;
					constraint = dimension == 1 ? Constraints.bounded(lower, upper)
							: Constraints.boundedVector(lower, upper, dimension);
					initial = new double[dimension]; Arrays.fill(initial, 0.5 * (lower + upper));
				} else if (declaration.lower != null) {
					double lower = declaration.lower.eval(constants).value;
					constraint = Constraints.lowerBound(lower, dimension);
					initial = new double[dimension]; Arrays.fill(initial, lower + 1.0);
				} else if (declaration.upper != null) {
					double upper = declaration.upper.eval(constants).value;
					constraint = Constraints.upperBound(upper, dimension);
					initial = new double[dimension]; Arrays.fill(initial, upper - 1.0);
				} else {
					constraint = dimension == 1 ? Constraints.real() : Constraints.realVector(dimension);
					initial = new double[dimension];
				}
				if (constraint.constrainedDimension() != dimension) {
					if (dimension % constraint.constrainedDimension() != 0)
						throw new ModelScriptException(Collections.singletonList(
								declaration.error("constraint shape does not tile the declared array")));
					int repetitions = dimension / constraint.constrainedDimension();
					constraint = Constraints.repeated(constraint, repetitions);
					double[] repeated = new double[dimension];
					for (int i = 0; i < repetitions; i++)
						System.arraycopy(initial, 0, repeated, i * initial.length, initial.length);
					initial = repeated;
				}
				builder.parameter(declaration.name, constraint, initial);
			}
			for (Assignment assignment : transformedParameters) {
				int[] assignmentShape = assignment.declaredShape(constants);
				shapes.put(assignment.name, assignmentShape); types.put(assignment.name, assignment.valueType());
			}
			for (Assignment assignment : generated) {
				int[] assignmentShape = assignment.declaredShape(constants);
				shapes.put(assignment.name, assignmentShape); types.put(assignment.name, assignment.valueType());
			}
			final int constrainedDimension = constrainedDimension(parameters, constants);
			final Set<String> declaredNames = new LinkedHashSet<String>();
			for (Declaration declaration : parameters) declaredNames.add(declaration.name);
			declaredNames.addAll(dataValues.keySet());
			boolean procedural = false;
			for (Statement statement : model) procedural |= statement.procedural();
			if (procedural) {
				addModelFactor(builder, "script:model", new BlockStatement(model, false),
						declaredNames, dataValues, shapes, types, constrainedDimension);
			} else {
				int statementIndex = 0;
				for (Statement statement : model)
					addModelFactor(builder, "script:" + (++statementIndex) + ":" + statement.label(),
							statement, declaredNames, dataValues, shapes, types, constrainedDimension);
			}
			final BayesianModel compiled = builder.build();
			try {
				compiled.logDensityAndGradient(compiled.initialState(), new double[compiled.dimension()]);
			} catch (IllegalArgumentException exception) {
				throw new ModelScriptException(Collections.singletonList(new ScriptDiagnostic(1, 1,
						"type/shape validation failed: " + exception.getMessage())));
			}
			final Map<String, double[]> capturedData = copyData(dataValues);
			final Map<String, int[]> capturedShapes = copyShapes(shapes);
			final Map<String, ValueType> capturedTypes = copyTypes(types);
			final List<Assignment> capturedTransforms = new ArrayList<Assignment>(transformedParameters);
			final List<Assignment> capturedGenerated = new ArrayList<Assignment>(generated);
			CompiledModelScript.Generator generator = new CompiledModelScript.Generator() {
				@Override public Map<String, double[]> generate(ModelState state, RandomEngine random) {
					Context context = new Context(state, capturedData, capturedShapes, capturedTypes,
							constrainedDimension, random, functions);
					for (Assignment assignment : capturedTransforms)
						context.setLocal(assignment.name, assignment.runtimeValue(context));
					Map<String, double[]> result = new LinkedHashMap<String, double[]>();
					for (Assignment assignment : capturedGenerated) {
						RuntimeValue value = assignment.runtimeValue(context);
						context.setLocal(assignment.name, value);
						result.put(assignment.name, values(value.values));
					}
					return result;
				}
			};
			return new CompiledModelScript(compiled, generator, LANGUAGE_VERSION);
		}

		private void addModelFactor(ModelBuilder builder, String factorName,
				final Statement statement, Set<String> declaredNames,
				Map<String, double[]> dataValues, Map<String, int[]> shapes,
				Map<String, ValueType> types,
				final int constrainedDimension) {
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
			final Map<String, int[]> capturedShapes = copyShapes(shapes);
			final Map<String, ValueType> capturedTypes = copyTypes(types);
			final List<Assignment> capturedTransforms =
					new ArrayList<Assignment>(transformedParameters);
			builder.factor(factorName, dependencies.toArray(new String[dependencies.size()]),
					new DifferentiableModelFactor() {
			private final ThreadLocal<ReverseTape> tapes = new ThreadLocal<ReverseTape>() {
				@Override protected ReverseTape initialValue() {
					return new ReverseTape(Math.max(1024, constrainedDimension * 32));
				}
			};
				@Override public double logDensityAndAddGradient(ModelState state,
						double[] gradient) {
					ReverseTape tape = tapes.get(); tape.reset();
					Diff.beginReverse(tape, constrainedDimension);
					try {
						Context context = new Context(state, capturedData, capturedShapes, capturedTypes,
								constrainedDimension, null, functions);
						for (Assignment assignment : capturedTransforms)
							context.setLocal(assignment.name, assignment.runtimeValue(context));
						Diff contribution = statement.eval(context).add(context.consumeFunctionTarget());
						Diff.reverseInto(contribution, gradient);
						return contribution.value;
					} finally { Diff.endReverse(); }
				}
			});
		}
	}

	private static final class Declaration {
		final String type; final String name; final Expr dimension;
		final List<Expr> shape; final Expr lower; final Expr upper;
		final List<Expr> baseShape; final int arrayRank;
		final Expr offset; final Expr multiplier; final boolean integer; final Token token;
		Declaration(String type, String name, Expr dimension, Expr lower,
				Expr upper, Expr offset, Expr multiplier, List<Expr> shape,
				List<Expr> baseShape, int arrayRank, boolean integer, Token token) {
			this.type = type; this.name = name; this.dimension = dimension;
			this.lower = lower; this.upper = upper; this.offset = offset;
			this.multiplier = multiplier; this.shape = shape; this.baseShape = baseShape;
			this.arrayRank = arrayRank;
			this.integer = integer; this.token = token;
		}
		ValueType valueType() { return new ValueType(valueKind(type), arrayRank); }
		int baseDimension(Context context, int index) {
			return checkedDimension(baseShape.get(index).eval(context).value, this);
		}
		ScriptDiagnostic error(String message) { return new ScriptDiagnostic(token.line, token.column, message); }
	}
	private static final class Assignment {
		final String name; final Expr expression; final List<Expr> shape; final Token token;
		final ValueKind kind; final int arrayRank;
		Assignment(String name, Expr expression, List<Expr> shape, ValueKind kind,
				int arrayRank, Token token) {
			this.name = name; this.expression = expression; this.shape = shape; this.kind = kind;
			this.arrayRank = arrayRank; this.token = token;
		}
		ValueType valueType() { return new ValueType(kind, arrayRank); }
		RuntimeValue runtimeValue(Context context) {
			RuntimeValue evaluated = promoteComplex(expression.evalValue(context), kind, context);
			return wrap(evaluated.values, checkedShape(context, evaluated.values.length));
		}
		RuntimeValue wrap(Diff[] values, int[] checkedShape) {
			return new RuntimeValue(values, checkedShape, kind, arrayRank);
		}
		int[] declaredShape(Context context) {
			if (shape.isEmpty()) return new int[0];
			int[] result = new int[shape.size()];
			for (int i = 0; i < result.length; i++) {
				double value = shape.get(i).eval(context).value;
				if (value != Math.rint(value) || value < 1 || value > 100000)
					throw new ModelScriptException(Collections.singletonList(
							new ScriptDiagnostic(token.line, token.column, "invalid assignment dimension")));
				result[i] = (int) value;
			}
			return result;
		}
		int[] checkedShape(Context context, int actualLength) {
			int width = storageWidth(kind);
			if (actualLength % width != 0) throw new ModelScriptException(Collections.singletonList(
					new ScriptDiagnostic(token.line, token.column, "initializer storage does not match type")));
			int logicalLength = actualLength / width;
			if (shape.isEmpty()) return logicalLength == 1 ? new int[0] : new int[] {logicalLength};
			int[] result = declaredShape(context); long count = 1;
			for (int extent : result) count *= extent;
			if (count != logicalLength) throw new ModelScriptException(Collections.singletonList(
					new ScriptDiagnostic(token.line, token.column,
							"initializer size does not match declaration")));
			return result;
		}
	}
	private static final class FunctionType {
		final ValueKind kind; final int arrayRank; final boolean integer; final TupleTypeSpec tuple;
		FunctionType(ValueKind kind, int arrayRank, boolean integer) {
			this(kind, arrayRank, integer, null);
		}
		FunctionType(TupleTypeSpec tuple) {
			this(ValueKind.TUPLE, 0, false, tuple);
		}
		private FunctionType(ValueKind kind, int arrayRank, boolean integer, TupleTypeSpec tuple) {
			this.kind = kind; this.arrayRank = arrayRank; this.integer = integer; this.tuple = tuple;
		}
		boolean sameSignature(FunctionType other) {
			return kind == other.kind && arrayRank == other.arrayRank && integer == other.integer
					&& sameTupleSignature(tuple, other.tuple);
		}
		boolean accepts(RuntimeValue value) {
			return kind == value.kind && arrayRank == value.arrayRank
					&& (tuple == null || acceptsTuple(tuple, value));
		}
		private static boolean acceptsTuple(TupleTypeSpec type, RuntimeValue value) {
			if (type.kind != value.kind || type.arrayRank != value.arrayRank) return false;
			if (type.kind != ValueKind.TUPLE) return true;
			if (value.tuple.length != type.members.length) return false;
			for (int i = 0; i < type.members.length; i++)
				if (!acceptsTuple(type.members[i], value.tuple[i])) return false;
			return true;
		}
		private static boolean sameTupleSignature(TupleTypeSpec left, TupleTypeSpec right) {
			if (left == null || right == null) return left == right;
			if (left.kind != right.kind || left.arrayRank != right.arrayRank) return false;
			if (left.kind != ValueKind.TUPLE) return true;
			if (left.members.length != right.members.length) return false;
			for (int i = 0; i < left.members.length; i++)
				if (!sameTupleSignature(left.members[i], right.members[i])) return false;
			return true;
		}
		String display() {
			if (tuple != null) return displayTuple(tuple);
			StringBuilder result = new StringBuilder();
			for (int i = 0; i < arrayRank; i++) result.append("array[] ");
			return result.append(integer ? "int" : kind == ValueKind.SCALAR ? "real"
					: kind.name().toLowerCase()).toString();
		}
		private static String displayTuple(TupleTypeSpec type) {
			if (type.kind != ValueKind.TUPLE) {
				StringBuilder result = new StringBuilder();
				for (int i = 0; i < type.arrayRank; i++) result.append("array[] ");
				return result.append(type.kind == ValueKind.SCALAR ? "real"
						: type.kind.name().toLowerCase()).toString();
			}
			StringBuilder result = new StringBuilder("tuple(");
			for (int i = 0; i < type.members.length; i++) {
				if (i > 0) result.append(", "); result.append(displayTuple(type.members[i]));
			}
			return result.append(')').toString();
		}
	}
	private static final class FunctionArgument {
		final String name; final FunctionType type; final boolean dataOnly;
		FunctionArgument(String name, FunctionType type, boolean dataOnly) {
			this.name = name; this.type = type; this.dataOnly = dataOnly;
		}
	}
	private static final class UserFunction {
		final String name; final FunctionType returnType; final FunctionArgument[] arguments;
		Statement body; StanExternalFunction external; final Token token;
		UserFunction(String name, FunctionType returnType, FunctionArgument[] arguments,
				Statement body, Token token) {
			this.name = name; this.returnType = returnType; this.arguments = arguments;
			this.body = body; this.token = token;
		}
		RuntimeValue invoke(RuntimeValue[] values, boolean[] dataArguments, Context context) {
			if (body == null) return invokeExternal(values);
			if (++context.functionDepth > 1000) {
				context.functionDepth--;
				throw new IllegalStateException("user-function recursion exceeded 1000 calls");
			}
			context.pushScope();
			context.functionNames.add(name);
			try {
				for (int i = 0; i < arguments.length; i++)
					context.declareLocal(arguments[i].name, values[i], arguments[i].type.integer,
							dataArguments[i]);
				try { body.eval(context); }
				catch (ReturnSignal returned) {
					if (!matchesReturn(returned.value)) throw new IllegalArgumentException("function '"
							+ name + "' returned " + returned.value.kind + Arrays.toString(returned.value.shape)
							+ " but declares " + returnType.display());
					return returned.value;
				}
				throw new IllegalArgumentException("function '" + name + "' did not return a value");
			} finally {
				context.functionNames.remove(context.functionNames.size() - 1);
				context.popScope(); context.functionDepth--;
			}
		}
		private RuntimeValue invokeExternal(RuntimeValue[] arguments) {
			if (external == null) throw new IllegalArgumentException("function '"+name+"' has no external binding");
			if (returnType.kind == ValueKind.TUPLE)
				throw new IllegalArgumentException("external tuple returns require a Java adapter function");
			double[][] plain = new double[arguments.length][]; int inputCount = 0;
			for (int i = 0; i < arguments.length; i++) {
				Diff[] flattened = flatten(arguments[i]); plain[i] = values(flattened); inputCount += flattened.length;
			}
			ExternalFunctionResult evaluated = external.evaluate(plain);
			if (evaluated == null) throw new IllegalArgumentException("external function '"+name+"' returned null");
			double[] output = evaluated.values(); int[] shape = evaluated.shape();
			double[][] jacobian = evaluated.jacobian(); Diff[] inputs = new Diff[inputCount]; int input = 0;
			for (RuntimeValue argument : arguments) for (Diff value : flatten(argument)) inputs[input++] = value;
			Diff[] result = new Diff[output.length];
			for (int row = 0; row < result.length; row++) {
				if (jacobian[row].length != inputCount)
					throw new IllegalArgumentException("external function '"+name+"' Jacobian column mismatch");
				result[row] = Diff.atomic(output[row], inputs, jacobian[row]);
			}
			if (returnType.arrayRank > shape.length)
				throw new IllegalArgumentException("external function '"+name+"' result rank mismatch");
			return new RuntimeValue(result, shape, returnType.kind, returnType.arrayRank);
		}
		private Diff[] flatten(RuntimeValue value) {
			if (value.kind != ValueKind.TUPLE) return value.values;
			List<Diff> result = new ArrayList<Diff>(); flatten(value, result);
			return result.toArray(new Diff[result.size()]);
		}
		private void flatten(RuntimeValue value, List<Diff> result) {
			if (value.kind != ValueKind.TUPLE) { result.addAll(Arrays.asList(value.values)); return; }
			for (RuntimeValue member : value.tuple) flatten(member, result);
		}
		private boolean matchesReturn(RuntimeValue value) {
			if (!returnType.accepts(value)) return false;
			if (returnType.integer) for (Diff element : value.values)
				if (element.value != Math.rint(element.value)) return false;
			return true;
		}
		boolean sameSignature(UserFunction other) {
			if (arguments.length != other.arguments.length) return false;
			for (int i = 0; i < arguments.length; i++)
				if (!arguments[i].type.sameSignature(other.arguments[i].type)) return false;
			return true;
		}
	}
	private static final class ReturnSignal extends RuntimeException {
		private static final long serialVersionUID = 1L; final transient RuntimeValue value;
		ReturnSignal(RuntimeValue value) { super(null, null, false, false); this.value = value; }
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
		@Override public Diff eval(Context context) {
			Diff contribution = expression.eval(context);
			if (context.functionDepth == 0) return contribution;
			context.addFunctionTarget(contribution);
			return Diff.constant(0.0, context.dimension);
		}
		@Override public void collect(Set<String> names) { expression.collect(names); }
		@Override public String label() { return "target"; }
	}
	private static final class ReturnStatement implements Statement {
		final Expr expression;
		ReturnStatement(Expr expression) { this.expression = expression; }
		@Override public Diff eval(Context context) { throw new ReturnSignal(expression.evalValue(context)); }
		@Override public void collect(Set<String> names) { expression.collect(names); }
		@Override public String label() { return "return"; }
		@Override public boolean procedural() { return true; }
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
		final List<Expr> shape; final ValueKind kind; final int arrayRank;
		LocalDeclarationStatement(String name, Expr initializer, boolean integer,
				List<Expr> shape, ValueKind kind, int arrayRank) {
			this.name = name; this.initializer = initializer; this.integer = integer;
			this.shape = shape; this.kind = kind; this.arrayRank = arrayRank;
		}
		@Override public Diff eval(Context context) {
			int[] extents = new int[shape.size()]; int logicalCount = 1;
			for (int i = 0; i < extents.length; i++) {
				double dimension = shape.get(i).eval(context).value;
				if (dimension != Math.rint(dimension) || dimension < 1 || dimension > 100000)
					throw new IllegalArgumentException("invalid local-container dimension for " + name);
				extents[i] = (int) dimension; logicalCount *= extents[i];
			}
			int count = logicalCount * storageWidth(kind);
			RuntimeValue value;
			if (initializer == null) {
				Diff[] uninitialized = new Diff[count];
				for (int i = 0; i < count; i++) uninitialized[i] = Diff.constant(Double.NaN, context.dimension);
				value = new RuntimeValue(uninitialized, extents, kind, arrayRank);
			} else {
				RuntimeValue evaluated = promoteComplex(initializer.evalValue(context), kind, context);
				if (evaluated.values.length != count)
					throw new IllegalArgumentException("initializer size does not match local declaration " + name);
				value = new RuntimeValue(evaluated.values, extents, kind, arrayRank);
			}
			if (integer) for (Diff element : value.values) if (element.value != Math.rint(element.value))
				throw new IllegalArgumentException("integer local '" + name + "' received a non-integer value");
			context.declareLocal(name, value, integer, initializer != null && initializer.dataOnly(context));
			return Diff.constant(0.0, context.dimension);
		}
		@Override public void collect(Set<String> names) {
			for (Expr extent : shape) extent.collect(names); if (initializer != null) initializer.collect(names);
		}
		@Override public String label() { return "local"; }
		@Override public boolean procedural() { return true; }
	}
	private static final class TupleTypeSpec {
		final ValueKind kind; final int arrayRank; final List<Expr> shape; final TupleTypeSpec[] members;
		TupleTypeSpec(ValueKind kind,int arrayRank,List<Expr> shape,TupleTypeSpec[] members){
			this.kind=kind;this.arrayRank=arrayRank;this.shape=shape;this.members=members;
		}
		void validate(RuntimeValue value,Context context,String name){
			if(kind==ValueKind.TUPLE){
				if(value.kind!=ValueKind.TUPLE||value.tuple.length!=members.length)
					throw new IllegalArgumentException("tuple initializer does not match "+name);
				for(int i=0;i<members.length;i++)members[i].validate(value.tuple[i],context,name+"."+(i+1));
				return;
			}
			if(value.kind!=kind||value.arrayRank!=arrayRank)
				throw new IllegalArgumentException("tuple member type mismatch for "+name);
			int[] expected=new int[shape.size()];
			for(int i=0;i<expected.length;i++)expected[i]=checkedLoopBound(shape.get(i).eval(context).value,name);
			if(!Arrays.equals(expected,value.shape))throw new IllegalArgumentException("tuple member shape mismatch for "+name);
		}
		void collect(Set<String> names){for(Expr extent:shape)extent.collect(names);if(members!=null)for(TupleTypeSpec member:members)member.collect(names);}
	}
	private static final class TupleLocalDeclarationStatement implements Statement {
		final String name;final TupleTypeSpec type;final Expr initializer;
		TupleLocalDeclarationStatement(String name,TupleTypeSpec type,Expr initializer){this.name=name;this.type=type;this.initializer=initializer;}
		@Override public Diff eval(Context context){
			if(initializer==null)throw new IllegalArgumentException("tuple local '"+name+"' requires an initializer");
			RuntimeValue value=initializer.evalValue(context);type.validate(value,context,name);
			context.declareLocal(name,value,false,initializer.dataOnly(context));return Diff.constant(0,context.dimension);
		}
		@Override public void collect(Set<String> names){type.collect(names);if(initializer!=null)initializer.collect(names);}
		@Override public String label(){return "tuple-local";}
		@Override public boolean procedural(){return true;}
	}
	private static final class TupleAssignmentStatement implements Statement {
		final TupleAccessExpr target;final Expr expression;
		TupleAssignmentStatement(TupleAccessExpr target,Expr expression){this.target=target;this.expression=expression;}
		@Override public Diff eval(Context context){
			List<Integer> path=new ArrayList<Integer>();Expr root=target;
			while(root instanceof TupleAccessExpr){TupleAccessExpr access=(TupleAccessExpr)root;path.add(0,access.index);root=access.tuple;}
			if(!(root instanceof VariableExpr)||!((VariableExpr)root).indices.isEmpty())
				throw new IllegalArgumentException("tuple assignment requires a local tuple variable");
			String name=((VariableExpr)root).name;RuntimeValue tuple=context.local(name);
			if(tuple==null||tuple.kind!=ValueKind.TUPLE)throw new IllegalArgumentException("unknown local tuple: "+name);
			RuntimeValue current=tuple;for(int index:path){
				if(current.kind!=ValueKind.TUPLE||index<0||index>=current.tuple.length)
					throw new IllegalArgumentException("tuple member index is out of range");
				current=current.tuple[index];
			}
			RuntimeValue value=expression.evalValue(context);
			value=promoteComplex(value,current.kind,context);
			if(value.kind!=current.kind||value.arrayRank!=current.arrayRank||!Arrays.equals(value.shape,current.shape))
				throw new IllegalArgumentException("tuple member assignment type mismatch");
			context.assignLocal(name,replace(tuple,path,0,value));return Diff.constant(0,context.dimension);
		}
		private RuntimeValue replace(RuntimeValue tuple,List<Integer> path,int depth,RuntimeValue value){
			RuntimeValue[] members=tuple.tuple.clone();int index=path.get(depth);
			members[index]=depth+1==path.size()?value:replace(members[index],path,depth+1,value);
			return RuntimeValue.tuple(members);
		}
		@Override public void collect(Set<String> names){target.collect(names);expression.collect(names);}
		@Override public String label(){return "tuple-assign";}
		@Override public boolean procedural(){return true;}
	}
	private static final class AssignmentStatement implements Statement {
		final VariableExpr target; final String operator; final Expr expression;
		AssignmentStatement(VariableExpr target, String operator, Expr expression) {
			this.target = target; this.operator = operator; this.expression = expression;
		}
		@Override public Diff eval(Context context) {
			RuntimeValue current = context.local(target.name);
			if (current == null) throw new IllegalArgumentException(
					"assignment requires a declared local variable: " + target.name);
			RuntimeValue value = promoteComplex(expression.evalValue(context), current.kind, context);
			if (target.indices.isEmpty()) {
				if (!operator.equals("=")) value = compound(current, value, context);
				context.assignLocal(target.name, value);
			} else {
				Selection selection = target.selection(context, current);
				int width = storageWidth(current.kind), logicalValues = value.values.length / width;
				if (logicalValues != 1 && logicalValues != selection.offsets.length)
					throw new IllegalArgumentException("indexed assignment shape mismatch for " + target.name);
				Diff[] replaced = current.values.clone();
				for (int i = 0; i < selection.offsets.length; i++) {
					int source = (logicalValues == 1 ? 0 : i)*width, targetOffset = selection.offsets[i]*width;
					if (operator.equals("=")) System.arraycopy(value.values,source,replaced,targetOffset,width);
					else if (width == 1) replaced[targetOffset] = compoundScalar(replaced[targetOffset],value.values[source]);
					else {
						Diff[] assigned = complexApply(operator.substring(0,1),replaced[targetOffset],
								replaced[targetOffset+1],value.values[source],value.values[source+1]);
						replaced[targetOffset]=assigned[0]; replaced[targetOffset+1]=assigned[1];
					}
				}
				context.assignLocal(target.name, new RuntimeValue(replaced, current.shape, current.kind, current.arrayRank));
			}
			return Diff.constant(0.0, context.dimension);
		}
		private RuntimeValue compound(RuntimeValue current, RuntimeValue value, Context context) {
			if (current.isComplex()) {
				int currentCount=current.logicalSize(), valueCount=value.logicalSize();
				if (valueCount != 1 && valueCount != currentCount)
					throw new IllegalArgumentException("compound assignment shape mismatch");
				Diff[] result = new Diff[current.values.length];
				for (int i=0;i<currentCount;i++) {
					int source=(valueCount==1?0:i)*2; Diff[] pair=complexApply(operator.substring(0,1),
							current.values[2*i],current.values[2*i+1],value.values[source],value.values[source+1]);
					result[2*i]=pair[0]; result[2*i+1]=pair[1];
				}
				return new RuntimeValue(result,current.shape,current.kind,current.arrayRank);
			}
			if (current.values.length != value.values.length && value.values.length != 1)
				throw new IllegalArgumentException("compound assignment shape mismatch");
			Diff[] result = new Diff[current.values.length];
			for (int i = 0; i < result.length; i++)
				result[i] = compoundScalar(current.values[i], value.values[value.values.length == 1 ? 0 : i]);
			return new RuntimeValue(result, current.shape, current.kind, current.arrayRank);
		}
		private Diff compoundScalar(Diff current, Diff value) {
			if (operator.equals("+=")) return current.add(value);
			if (operator.equals("-=")) return current.subtract(value);
			if (operator.equals("*=")) return current.multiply(value);
			if (operator.equals("/=")) return current.divide(value);
			throw new IllegalStateException("unknown assignment operator " + operator);
		}
		@Override public void collect(Set<String> names) { target.collect(names); expression.collect(names); }
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
				context.declareLocal(variable, RuntimeValue.scalar(Diff.constant(from, context.dimension)), true, true);
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
			if (context.functions != null) {
				String density = distribution + "_lpdf", mass = distribution + "_lpmf";
				String selected = context.functions.containsKey(density) ? density
						: context.functions.containsKey(mass) ? mass : null;
				if (selected != null) {
					List<Expr> values = new ArrayList<Expr>(); values.add(left); values.addAll(arguments);
					return capture(new CallExpr(selected, values).eval(context), context);
				}
			}
			if (distribution.equals("multi_normal") || distribution.equals("multi_normal_cholesky")) {
				List<Expr> values = new ArrayList<Expr>(); values.add(left); values.addAll(arguments);
				return capture(new CallExpr(distribution + "_lpdf", values).eval(context), context);
			}
			RuntimeValue observationValue = left.evalValue(context);
			Diff[] observations = observationValue.values;
			RuntimeValue[] evaluatedArguments = new RuntimeValue[arguments.size()];
			RuntimeValue template = observationValue.isScalar() ? null : observationValue;
			int vectorizedLength = observations.length;
			for (int i = 0; i < arguments.size(); i++) {
				evaluatedArguments[i] = arguments.get(i).evalValue(context);
				if (!evaluatedArguments[i].isScalar()) {
					if (template != null && template.values.length != evaluatedArguments[i].values.length)
						throw new IllegalArgumentException("incompatible distribution broadcast: "
								+ template.kind + Arrays.toString(template.shape) + " versus "
								+ evaluatedArguments[i].kind + Arrays.toString(evaluatedArguments[i].shape));
					template = evaluatedArguments[i]; vectorizedLength = evaluatedArguments[i].values.length;
				}
			}
			Diff result = Diff.constant(0.0, context.dimension);
			for (int element = 0; element < vectorizedLength; element++) {
				Diff[] values = new Diff[arguments.size() + 1];
				values[0] = observations[observations.length == 1 ? 0 : element];
				for (int i = 0; i < arguments.size(); i++) {
					RuntimeValue argument = evaluatedArguments[i];
					values[i + 1] = argument.values[argument.isScalar() ? 0 : element];
				}
				result = result.add(logProbability(distribution, values));
			}
			return capture(result, context);
		}
		private Diff capture(Diff contribution, Context context) {
			if (context.functionDepth == 0) return contribution;
			context.addFunctionTarget(contribution);
			return Diff.constant(0.0, context.dimension);
		}
		@Override public void collect(Set<String> names) {
			left.collect(names); for (Expr argument : arguments) argument.collect(names);
		}
		@Override public String label() { return distribution; }
	}

	private enum ValueKind {
		SCALAR, VECTOR, ROW_VECTOR, MATRIX,
		COMPLEX, COMPLEX_VECTOR, COMPLEX_ROW_VECTOR, COMPLEX_MATRIX, TUPLE
	}
	private static final class RuntimeValue {
		final Diff[] values; final int[] shape; final ValueKind kind; final int arrayRank;
		final RuntimeValue[] tuple;
		RuntimeValue(Diff[] values, int[] shape, ValueKind kind, int arrayRank) {
			this.values = values; this.shape = shape.clone(); this.kind = kind; this.arrayRank = arrayRank; this.tuple = null;
			long count = 1; for (int extent : shape) count *= extent;
			if (shape.length == 0) count = 1;
			if (count * storageWidth(kind) != values.length || arrayRank < 0 || arrayRank > shape.length)
				throw new IllegalArgumentException("invalid runtime value shape");
		}
		private RuntimeValue(RuntimeValue[] tuple) {
			if (tuple == null || tuple.length == 0) throw new IllegalArgumentException("tuple must not be empty");
			this.tuple = tuple.clone(); this.values = new Diff[0]; this.shape = new int[] {tuple.length};
			this.kind = ValueKind.TUPLE; this.arrayRank = 0;
		}
		static RuntimeValue scalar(Diff value) {
			return new RuntimeValue(new Diff[] {value}, new int[0], ValueKind.SCALAR, 0);
		}
		static RuntimeValue complex(Diff real, Diff imaginary) {
			return new RuntimeValue(new Diff[] {real, imaginary}, new int[0], ValueKind.COMPLEX, 0);
		}
		static RuntimeValue tuple(RuntimeValue[] values) { return new RuntimeValue(values); }
		Diff scalar(String description) {
			if (kind != ValueKind.SCALAR || values.length != 1)
				throw new IllegalArgumentException(description + " must be a real scalar");
			return values[0];
		}
		boolean isScalar() { return shape.length == 0 && kind != ValueKind.TUPLE; }
		boolean isComplex() { return complexKind(kind); }
		int logicalSize() { return kind == ValueKind.TUPLE ? tuple.length : values.length / storageWidth(kind); }
		int rows() {
			if (kind != ValueKind.MATRIX && kind != ValueKind.COMPLEX_MATRIX || shape.length < 2)
				throw new IllegalArgumentException("matrix required");
			return shape[shape.length - 2];
		}
		int columns() {
			if (kind != ValueKind.MATRIX && kind != ValueKind.COMPLEX_MATRIX || shape.length < 2)
				throw new IllegalArgumentException("matrix required");
			return shape[shape.length - 1];
		}
	}
	private static final class ValueType {
		final ValueKind kind; final int arrayRank;
		ValueType(ValueKind kind, int arrayRank) { this.kind = kind; this.arrayRank = arrayRank; }
	}

	private interface Expr {
		RuntimeValue evalValue(Context context);
		default Diff eval(Context context) { return evalValue(context).scalar("expression"); }
		default Diff[] evalVector(Context context) { return evalValue(context).values; }
		default boolean dataOnly(Context context) { return false; }
		void collect(Set<String> names);
	}
	private static final class NumberExpr implements Expr {
		final double value; final boolean imaginary;
		NumberExpr(double value) { this(value, false); }
		NumberExpr(double value, boolean imaginary) { this.value = value; this.imaginary = imaginary; }
		@Override public RuntimeValue evalValue(Context context) {
			Diff number = Diff.constant(value, context.dimension);
			return imaginary ? RuntimeValue.complex(Diff.constant(0, context.dimension), number)
					: RuntimeValue.scalar(number);
		}
		@Override public boolean dataOnly(Context context) { return true; }
		@Override public void collect(Set<String> names) {}
	}
	private static final class LiteralExpr implements Expr {
		final List<Expr> elements; final boolean array;
		LiteralExpr(List<Expr> elements, boolean array) { this.elements = elements; this.array = array; }
		@Override public RuntimeValue evalValue(Context context) {
			if (elements.isEmpty())
				return new RuntimeValue(new Diff[0], new int[] {0}, ValueKind.SCALAR, array ? 1 : 0);
			RuntimeValue first = elements.get(0).evalValue(context);
			List<RuntimeValue> evaluated = new ArrayList<RuntimeValue>(); evaluated.add(first);
			for (int i = 1; i < elements.size(); i++) {
				RuntimeValue value = elements.get(i).evalValue(context);
				if (complexKind(value.kind) && !complexKind(first.kind)
						&& complexKindFor(first.kind) == value.kind) {
					first = promoteComplex(first, value.kind, context);
					for (int prior = 0; prior < evaluated.size(); prior++)
						evaluated.set(prior, promoteComplex(evaluated.get(prior), value.kind, context));
				}
				value = promoteComplex(value, first.kind, context);
				if (value.kind != first.kind || value.arrayRank != first.arrayRank
						|| !Arrays.equals(value.shape, first.shape))
					throw new IllegalArgumentException("literal elements must have one common type and shape");
				evaluated.add(value);
			}
			int total = 0; for (RuntimeValue value : evaluated) total += value.values.length;
			Diff[] values = new Diff[total]; int offset = 0;
			for (RuntimeValue value : evaluated) {
				System.arraycopy(value.values, 0, values, offset, value.values.length); offset += value.values.length;
			}
			if (array) {
				int[] shape = prepend(elements.size(), first.shape);
				return new RuntimeValue(values, shape, first.kind, first.arrayRank + 1);
			}
			if (first.isScalar())
				return new RuntimeValue(values, new int[] {elements.size()}, first.isComplex()
						? ValueKind.COMPLEX_ROW_VECTOR : ValueKind.ROW_VECTOR, 0);
			if ((first.kind == ValueKind.ROW_VECTOR || first.kind == ValueKind.COMPLEX_ROW_VECTOR)
					&& first.arrayRank == 0) {
				return new RuntimeValue(values, new int[] {elements.size(), first.logicalSize()},
						first.isComplex() ? ValueKind.COMPLEX_MATRIX : ValueKind.MATRIX, 0);
			}
			throw new IllegalArgumentException("bracket literals contain scalars or equal row vectors");
		}
		private int[] prepend(int extent, int[] shape) {
			int[] result = new int[shape.length + 1]; result[0] = extent;
			System.arraycopy(shape, 0, result, 1, shape.length); return result;
		}
		@Override public boolean dataOnly(Context context) {
			for (Expr element : elements) if (!element.dataOnly(context)) return false;
			return true;
		}
		@Override public void collect(Set<String> names) {
			for (Expr element : elements) element.collect(names);
		}
	}
	private static final class TupleExpr implements Expr {
		final List<Expr> elements;
		TupleExpr(List<Expr> elements){this.elements=elements;}
		@Override public RuntimeValue evalValue(Context context){
			RuntimeValue[] values=new RuntimeValue[elements.size()];
			for(int i=0;i<values.length;i++)values[i]=elements.get(i).evalValue(context);
			return RuntimeValue.tuple(values);
		}
		@Override public boolean dataOnly(Context context){for(Expr element:elements)if(!element.dataOnly(context))return false;return true;}
		@Override public void collect(Set<String> names){for(Expr element:elements)element.collect(names);}
	}
	private static final class TupleAccessExpr implements Expr {
		final Expr tuple; final int index;
		TupleAccessExpr(Expr tuple,int index){this.tuple=tuple;this.index=index;}
		@Override public RuntimeValue evalValue(Context context){
			RuntimeValue value=tuple.evalValue(context);
			if(value.kind!=ValueKind.TUPLE||index<0||index>=value.tuple.length)
				throw new IllegalArgumentException("tuple member index is out of range");
			return value.tuple[index];
		}
		@Override public boolean dataOnly(Context context){return tuple.dataOnly(context);}
		@Override public void collect(Set<String> names){tuple.collect(names);}
	}
	private static final class IndexSpec {
		final Expr lower; final Expr upper; final boolean single;
		private IndexSpec(Expr lower, Expr upper, boolean single) {
			this.lower = lower; this.upper = upper; this.single = single;
		}
		static IndexSpec single(Expr expression) { return new IndexSpec(expression, expression, true); }
		static IndexSpec range(Expr lower, Expr upper) { return new IndexSpec(lower, upper, false); }
		void collect(Set<String> names) {
			if (lower != null) lower.collect(names); if (upper != null && upper != lower) upper.collect(names);
		}
	}
	private static final class Selection {
		final int[] offsets; final int[] shape; final ValueKind kind; final int arrayRank;
		Selection(int[] offsets, int[] shape, ValueKind kind, int arrayRank) {
			this.offsets = offsets; this.shape = shape; this.kind = kind; this.arrayRank = arrayRank;
		}
	}
	private static final class VariableExpr implements Expr {
		final String name; final List<IndexSpec> indices;
		VariableExpr(String name, List<IndexSpec> indices) { this.name = name; this.indices = indices; }
		@Override public RuntimeValue evalValue(Context context) {
			RuntimeValue resolved = resolveValue(context);
			if (indices.isEmpty()) return resolved;
			Selection selection = selection(context, resolved);
			int width = storageWidth(resolved.kind); Diff[] selected = new Diff[selection.offsets.length*width];
			for (int i = 0; i < selection.offsets.length; i++)
				System.arraycopy(resolved.values,selection.offsets[i]*width,selected,i*width,width);
			return new RuntimeValue(selected, selection.shape, selection.kind, selection.arrayRank);
		}
		Selection selection(Context context, RuntimeValue resolved) {
			if (indices.size() > resolved.shape.length)
				throw new IllegalArgumentException("too many indexes for " + name + ": rank is " + resolved.shape.length);
			int rank = resolved.shape.length;
			int[][] coordinates = new int[rank][]; boolean[] retained = new boolean[rank];
			for (int dimension = 0; dimension < rank; dimension++) {
				IndexSpec spec = dimension < indices.size() ? indices.get(dimension) : IndexSpec.range(null, null);
				int extent = resolved.shape[dimension];
				if (spec.single) {
					coordinates[dimension] = new int[] {checkedIndex(spec.lower.eval(context).value, extent, name)};
				} else {
					int lower = spec.lower == null ? 0 : checkedIndex(spec.lower.eval(context).value, extent, name);
					int upper = spec.upper == null ? extent - 1 : checkedIndex(spec.upper.eval(context).value, extent, name);
					if (upper < lower) throw new IllegalArgumentException("range index for " + name + " has descending bounds");
					coordinates[dimension] = new int[upper - lower + 1]; retained[dimension] = true;
					for (int i = 0; i < coordinates[dimension].length; i++) coordinates[dimension][i] = lower + i;
				}
			}
			int count = 1, outputRank = 0, outputArrayRank = 0;
			for (int dimension = 0; dimension < rank; dimension++) {
				count *= coordinates[dimension].length;
				if (retained[dimension]) { outputRank++; if (dimension < resolved.arrayRank) outputArrayRank++; }
			}
			int[] offsets = new int[count], resultShape = new int[outputRank]; int shapeIndex = 0;
			for (int dimension = 0; dimension < rank; dimension++)
				if (retained[dimension]) resultShape[shapeIndex++] = coordinates[dimension].length;
			for (int output = 0; output < count; output++) {
				int selector = output, offset = 0;
				for (int dimension = rank - 1; dimension >= 0; dimension--) {
					int length = coordinates[dimension].length;
					int chosen = coordinates[dimension][selector % length]; selector /= length;
					int stride = 1; for (int later = dimension + 1; later < rank; later++) stride *= resolved.shape[later];
					offset += chosen * stride;
				}
				offsets[output] = offset;
			}
			ValueKind resultKind = selectedKind(resolved, retained);
			return new Selection(offsets, resultShape, resultKind, outputArrayRank);
		}
		private ValueKind selectedKind(RuntimeValue value, boolean[] retained) {
			int baseStart = value.arrayRank;
			if (value.kind == ValueKind.MATRIX || value.kind == ValueKind.COMPLEX_MATRIX) {
				boolean rows = retained[baseStart], columns = retained[baseStart + 1];
				boolean complex = value.kind == ValueKind.COMPLEX_MATRIX;
				if (rows && columns) return complex ? ValueKind.COMPLEX_MATRIX : ValueKind.MATRIX;
				if (rows) return complex ? ValueKind.COMPLEX_VECTOR : ValueKind.VECTOR;
				if (columns) return complex ? ValueKind.COMPLEX_ROW_VECTOR : ValueKind.ROW_VECTOR;
				return complex ? ValueKind.COMPLEX : ValueKind.SCALAR;
			}
			if ((value.kind == ValueKind.VECTOR || value.kind == ValueKind.ROW_VECTOR
					|| value.kind == ValueKind.COMPLEX_VECTOR || value.kind == ValueKind.COMPLEX_ROW_VECTOR)
					&& retained[baseStart]) return value.kind;
			return value.isComplex() ? ValueKind.COMPLEX : ValueKind.SCALAR;
		}
		private RuntimeValue resolveValue(Context context) {
			RuntimeValue local = context.local(name);
			if (local != null) return local;
			if (context.state != null && context.state.hasParameter(name)) {
				int size = context.state.parameterDimension(name);
				Diff[] result = new Diff[size];
				for (int i = 0; i < size; i++) {
					result[i] = Diff.variable(context.state.value(name, i), context.dimension,
							context.state.constrainedOffset(name) + i);
				}
				return context.wrap(name, result);
			}
			double[] values = context.data.get(name);
			if (values == null) throw new IllegalArgumentException("unknown variable: " + name);
			Diff[] result = new Diff[values.length];
			for (int i = 0; i < values.length; i++) result[i] = Diff.constant(values[i], context.dimension);
			return context.wrap(name, result);
		}
		private Diff[] resolveVector(Context context) { return resolveValue(context).values; }
		@Override public void collect(Set<String> names) {
			names.add(name); for (IndexSpec index : indices) index.collect(names);
		}
		@Override public boolean dataOnly(Context context) {
			if (!context.isDataVariable(name)) return false;
			for (IndexSpec index : indices) {
				if (index.lower != null && !index.lower.dataOnly(context)) return false;
				if (index.upper != null && !index.upper.dataOnly(context)) return false;
			}
			return true;
		}
	}
	private static final class UnaryExpr implements Expr {
		final String operator; final Expr operand;
		UnaryExpr(String operator, Expr operand) { this.operator = operator; this.operand = operand; }
		@Override public RuntimeValue evalValue(Context context) {
			RuntimeValue input = operand.evalValue(context); Diff[] values = input.values;
			Diff[] result = new Diff[values.length];
			for (int i = 0; i < values.length; i++) result[i] = apply(values[i], context);
			return new RuntimeValue(result, input.shape, input.kind, input.arrayRank);
		}
		private Diff apply(Diff value, Context context) {
			if (operator.equals("-")) return value.negate();
			if (operator.equals("!")) return booleanValue(!truth(value), context);
			return value;
		}
		@Override public void collect(Set<String> names) { operand.collect(names); }
		@Override public boolean dataOnly(Context context) { return operand.dataOnly(context); }
	}
	private static final class TransposeExpr implements Expr {
		final Expr operand;
		TransposeExpr(Expr operand) { this.operand = operand; }
		@Override public RuntimeValue evalValue(Context context) {
			RuntimeValue value = operand.evalValue(context);
			if (value.kind == ValueKind.VECTOR || value.kind == ValueKind.COMPLEX_VECTOR) {
				Diff[] values = conjugateIfComplex(value);
				return new RuntimeValue(values, value.shape, value.kind == ValueKind.COMPLEX_VECTOR
						? ValueKind.COMPLEX_ROW_VECTOR : ValueKind.ROW_VECTOR, value.arrayRank);
			}
			if (value.kind == ValueKind.ROW_VECTOR || value.kind == ValueKind.COMPLEX_ROW_VECTOR) {
				Diff[] values = conjugateIfComplex(value);
				return new RuntimeValue(values, value.shape, value.kind == ValueKind.COMPLEX_ROW_VECTOR
						? ValueKind.COMPLEX_VECTOR : ValueKind.VECTOR, value.arrayRank);
			}
			if (value.kind != ValueKind.MATRIX && value.kind != ValueKind.COMPLEX_MATRIX) return value;
			int rows = value.rows(), columns = value.columns();
			int width=storageWidth(value.kind), matrices = value.logicalSize() / (rows * columns);
			Diff[] result = new Diff[value.values.length];
			for (int matrix = 0; matrix < matrices; matrix++) {
				int base = matrix * rows * columns;
				for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
					for(int component=0;component<width;component++) {
						Diff element=value.values[(base+row*columns+column)*width+component];
						result[(base+column*rows+row)*width+component]=width==2&&component==1?element.negate():element;
					}
			}
			int[] shape = value.shape.clone(); shape[shape.length - 2] = columns; shape[shape.length - 1] = rows;
			return new RuntimeValue(result, shape, value.kind, value.arrayRank);
		}
		private Diff[] conjugateIfComplex(RuntimeValue value) {
			if(!value.isComplex()) return value.values;
			Diff[] result=value.values.clone(); for(int i=1;i<result.length;i+=2) result[i]=result[i].negate(); return result;
		}
		@Override public void collect(Set<String> names) { operand.collect(names); }
		@Override public boolean dataOnly(Context context) { return operand.dataOnly(context); }
	}
	private static final class BinaryExpr implements Expr {
		final String operator; final Expr left; final Expr right;
		BinaryExpr(String operator, Expr left, Expr right) {
			this.operator = operator; this.left = left; this.right = right;
		}
		@Override public RuntimeValue evalValue(Context context) {
			RuntimeValue leftValue = left.evalValue(context);
			if (operator.equals("&&") && !truth(leftValue.scalar("logical operand")))
				return RuntimeValue.scalar(booleanValue(false, context));
			if (operator.equals("||") && truth(leftValue.scalar("logical operand")))
				return RuntimeValue.scalar(booleanValue(true, context));
			RuntimeValue rightValue = right.evalValue(context);
			if (leftValue.isComplex() || rightValue.isComplex())
				return complexBinary(leftValue,rightValue,context);
			if (operator.equals("*") && !leftValue.isScalar() && !rightValue.isScalar())
				return matrixMultiply(leftValue, rightValue, context);
			if (operator.equals("/") && !leftValue.isScalar() && !rightValue.isScalar())
				throw new IllegalArgumentException("container right division is not supported; use mdivide_right");
			Diff[] a = leftValue.values, b = rightValue.values;
			int length = Math.max(a.length, b.length);
			if (a.length != 1 && b.length != 1 && (!Arrays.equals(leftValue.shape, rightValue.shape)
					|| leftValue.kind != rightValue.kind || leftValue.arrayRank != rightValue.arrayRank))
				throw new IllegalArgumentException("incompatible container shapes "
						+ leftValue.kind + Arrays.toString(leftValue.shape) + " and "
						+ rightValue.kind + Arrays.toString(rightValue.shape));
			Diff[] result = new Diff[length];
			for (int i = 0; i < length; i++)
				result[i] = apply(a[a.length == 1 ? 0 : i], b[b.length == 1 ? 0 : i], context);
			RuntimeValue template = a.length == 1 ? rightValue : leftValue;
			return length == 1 ? RuntimeValue.scalar(result[0])
					: new RuntimeValue(result, template.shape, template.kind, template.arrayRank);
		}
		private RuntimeValue complexBinary(RuntimeValue leftValue, RuntimeValue rightValue, Context context) {
			if(!leftValue.isComplex()) leftValue=promoteComplex(leftValue,complexKindFor(leftValue.kind),context);
			if(!rightValue.isComplex()) rightValue=promoteComplex(rightValue,complexKindFor(rightValue.kind),context);
			if(operator.equals("*")&&!leftValue.isScalar()&&!rightValue.isScalar())
				return complexMatrixMultiply(leftValue,rightValue,context);
			int leftCount=leftValue.logicalSize(), rightCount=rightValue.logicalSize();
			if(leftCount!=1&&rightCount!=1&&(!Arrays.equals(leftValue.shape,rightValue.shape)
					||leftValue.kind!=rightValue.kind||leftValue.arrayRank!=rightValue.arrayRank))
				throw new IllegalArgumentException("incompatible complex container shapes");
			int count=Math.max(leftCount,rightCount); boolean comparison=operator.equals("==")||operator.equals("!=");
			Diff[] result=new Diff[count*(comparison?1:2)];
			for(int i=0;i<count;i++) {
				int li=(leftCount==1?0:i)*2,ri=(rightCount==1?0:i)*2;
				if(comparison) {
					boolean equal=leftValue.values[li].value==rightValue.values[ri].value
							&&leftValue.values[li+1].value==rightValue.values[ri+1].value;
					result[i]=booleanValue(operator.equals("==")?equal:!equal,context);
				} else {
					Diff[] pair=complexApply(operator,leftValue.values[li],leftValue.values[li+1],
							rightValue.values[ri],rightValue.values[ri+1]); result[2*i]=pair[0];result[2*i+1]=pair[1];
				}
			}
			RuntimeValue template=leftCount==1?rightValue:leftValue;
			if(comparison) return count==1?RuntimeValue.scalar(result[0]):new RuntimeValue(result,template.shape,
					realKindFor(template.kind),template.arrayRank);
			return new RuntimeValue(result,template.shape,template.kind,template.arrayRank);
		}
		private RuntimeValue complexMatrixMultiply(RuntimeValue a,RuntimeValue b,Context context) {
			if(a.arrayRank>0||b.arrayRank>0) throw new IllegalArgumentException("complex matrix multiplication does not broadcast arrays");
			ValueKind ar=realKindFor(a.kind),br=realKindFor(b.kind); int rows,inner,columns; ValueKind resultKind;
			if(ar==ValueKind.ROW_VECTOR&&br==ValueKind.VECTOR) { rows=1;inner=a.logicalSize();columns=1;resultKind=ValueKind.COMPLEX; }
			else if(ar==ValueKind.VECTOR&&br==ValueKind.ROW_VECTOR) { rows=a.logicalSize();inner=1;columns=b.logicalSize();resultKind=ValueKind.COMPLEX_MATRIX; }
			else if(ar==ValueKind.MATRIX&&br==ValueKind.VECTOR) { rows=a.rows();inner=a.columns();columns=1;resultKind=ValueKind.COMPLEX_VECTOR; }
			else if(ar==ValueKind.ROW_VECTOR&&br==ValueKind.MATRIX) { rows=1;inner=a.logicalSize();columns=b.columns();resultKind=ValueKind.COMPLEX_ROW_VECTOR; }
			else if(ar==ValueKind.MATRIX&&br==ValueKind.MATRIX) { rows=a.rows();inner=a.columns();columns=b.columns();resultKind=ValueKind.COMPLEX_MATRIX; }
			else throw new IllegalArgumentException("unsupported complex matrix product");
			int bRows=br==ValueKind.MATRIX?b.rows():br==ValueKind.VECTOR?b.logicalSize():1;
			if(inner!=bRows) throw new IllegalArgumentException("complex matrix inner dimensions differ");
			Diff[] result=new Diff[rows*columns*2];
			for(int row=0;row<rows;row++)for(int column=0;column<columns;column++) {
				Diff real=Diff.constant(0,context.dimension),imag=Diff.constant(0,context.dimension);
				for(int k=0;k<inner;k++) {
					int ai=(ar==ValueKind.ROW_VECTOR?k:ar==ValueKind.VECTOR?row:row*inner+k)*2;
					int bi=(br==ValueKind.VECTOR?k:br==ValueKind.ROW_VECTOR?column:k*columns+column)*2;
					Diff[] pair=complexApply("*",a.values[ai],a.values[ai+1],b.values[bi],b.values[bi+1]);
					real=real.add(pair[0]);imag=imag.add(pair[1]);
				}
				result[(row*columns+column)*2]=real;result[(row*columns+column)*2+1]=imag;
			}
			int[] shape=resultKind==ValueKind.COMPLEX?new int[0]
					:resultKind==ValueKind.COMPLEX_MATRIX?new int[]{rows,columns}
					:new int[]{resultKind==ValueKind.COMPLEX_VECTOR?rows:columns};
			return new RuntimeValue(result,shape,resultKind,0);
		}
		private RuntimeValue matrixMultiply(RuntimeValue a, RuntimeValue b, Context context) {
			if (a.arrayRank > 0 || b.arrayRank > 0)
				throw new IllegalArgumentException("matrix multiplication does not broadcast across arrays");
			if (a.kind == ValueKind.ROW_VECTOR && b.kind == ValueKind.VECTOR) {
				if (a.values.length != b.values.length) throw shapeError(a, b, "row_vector * vector");
				Diff sum = Diff.constant(0.0, context.dimension);
				for (int i = 0; i < a.values.length; i++) sum = sum.add(a.values[i].multiply(b.values[i]));
				return RuntimeValue.scalar(sum);
			}
			if (a.kind == ValueKind.VECTOR && b.kind == ValueKind.ROW_VECTOR) {
				Diff[] result = new Diff[a.values.length * b.values.length];
				for (int row = 0; row < a.values.length; row++) for (int column = 0; column < b.values.length; column++)
					result[row * b.values.length + column] = a.values[row].multiply(b.values[column]);
				return new RuntimeValue(result, new int[] {a.values.length, b.values.length}, ValueKind.MATRIX, 0);
			}
			if (a.kind == ValueKind.MATRIX && b.kind == ValueKind.VECTOR) {
				if (a.columns() != b.values.length) throw shapeError(a, b, "matrix * vector");
				Diff[] result = multiply(a.values, a.rows(), a.columns(), b.values, b.values.length, 1, context);
				return new RuntimeValue(result, new int[] {a.rows()}, ValueKind.VECTOR, 0);
			}
			if (a.kind == ValueKind.ROW_VECTOR && b.kind == ValueKind.MATRIX) {
				if (a.values.length != b.rows()) throw shapeError(a, b, "row_vector * matrix");
				Diff[] result = multiply(a.values, 1, a.values.length, b.values, b.rows(), b.columns(), context);
				return new RuntimeValue(result, new int[] {b.columns()}, ValueKind.ROW_VECTOR, 0);
			}
			if (a.kind == ValueKind.MATRIX && b.kind == ValueKind.MATRIX) {
				if (a.columns() != b.rows()) throw shapeError(a, b, "matrix * matrix");
				Diff[] result = multiply(a.values, a.rows(), a.columns(), b.values, b.rows(), b.columns(), context);
				return new RuntimeValue(result, new int[] {a.rows(), b.columns()}, ValueKind.MATRIX, 0);
			}
			throw shapeError(a, b, "operator *");
		}
		private Diff[] multiply(Diff[] a, int ar, int ac, Diff[] b, int br, int bc, Context context) {
			Diff[] result = new Diff[ar * bc];
			for (int row = 0; row < ar; row++) for (int column = 0; column < bc; column++) {
				Diff sum = Diff.constant(0.0, context.dimension);
				for (int k = 0; k < ac; k++) sum = sum.add(a[row * ac + k].multiply(b[k * bc + column]));
				result[row * bc + column] = sum;
			}
			return result;
		}
		private IllegalArgumentException shapeError(RuntimeValue a, RuntimeValue b, String operation) {
			return new IllegalArgumentException(operation + " has incompatible shapes "
					+ Arrays.toString(a.shape) + " and " + Arrays.toString(b.shape));
		}
		private Diff apply(Diff a, Diff b, Context context) {
			if (operator.equals("&&") && !truth(a)) return booleanValue(false, context);
			if (operator.equals("||") && truth(a)) return booleanValue(true, context);
			if (operator.equals("+")) return a.add(b);
			if (operator.equals("-")) return a.subtract(b);
			if (operator.equals("*") || operator.equals(".*")) return a.multiply(b);
			if (operator.equals("/") || operator.equals("./")) return a.divide(b);
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
		@Override public boolean dataOnly(Context context) {
			return left.dataOnly(context) && right.dataOnly(context);
		}
	}
	private static final class ConditionalExpr implements Expr {
		final Expr condition; final Expr whenTrue; final Expr whenFalse;
		ConditionalExpr(Expr condition, Expr whenTrue, Expr whenFalse) {
			this.condition = condition; this.whenTrue = whenTrue; this.whenFalse = whenFalse;
		}
		@Override public RuntimeValue evalValue(Context context) {
			return truth(condition.eval(context)) ? whenTrue.evalValue(context) : whenFalse.evalValue(context);
		}
		@Override public void collect(Set<String> names) {
			condition.collect(names); whenTrue.collect(names); whenFalse.collect(names);
		}
		@Override public boolean dataOnly(Context context) {
			return condition.dataOnly(context) && whenTrue.dataOnly(context) && whenFalse.dataOnly(context);
		}
	}
	private static final class CallExpr implements Expr {
		final String name; final List<Expr> arguments;
		CallExpr(String name, List<Expr> arguments) { this.name = name; this.arguments = arguments; }
		@Override public RuntimeValue evalValue(Context context) {
			RuntimeValue structured = structuredFunction(context);
			if (structured != null) return structured;
			RuntimeValue userDefined = callUserFunction(context);
			if (userDefined != null) return userDefined;
			RuntimeValue broadcast = broadcastScalarFunction(context);
			if (broadcast != null) return broadcast;
			if ((name.equals("to_vector") || name.equals("to_row_vector")
					|| name.equals("to_array_1d")) && arguments.size() == 1) {
				RuntimeValue input = arguments.get(0).evalValue(context);
				ValueKind kind = name.equals("to_vector") ? ValueKind.VECTOR
						: name.equals("to_row_vector") ? ValueKind.ROW_VECTOR : ValueKind.SCALAR;
				return new RuntimeValue(input.values, new int[] {input.values.length}, kind,
						name.equals("to_array_1d") ? 1 : 0);
			}
			if ((name.equals("rep_vector") || name.equals("rep_row_vector")
					|| name.equals("rep_array")) && arguments.size() == 2) {
				Diff[] result = evalVector(context);
				ValueKind kind = name.equals("rep_vector") ? ValueKind.VECTOR
						: name.equals("rep_row_vector") ? ValueKind.ROW_VECTOR : ValueKind.SCALAR;
				return new RuntimeValue(result, new int[] {result.length}, kind,
						name.equals("rep_array") ? 1 : 0);
			}
			if (arguments.size() == 1 && !name.endsWith("_rng")) {
				RuntimeValue input = arguments.get(0).evalValue(context);
				if (input.values.length > 1) {
					Diff[] result = evalVector(context);
					if (result.length == input.values.length)
						return new RuntimeValue(result, input.shape, input.kind, input.arrayRank);
				}
			}
			return RuntimeValue.scalar(eval(context));
		}
		private RuntimeValue broadcastScalarFunction(Context context) {
			if (name.equals("sum") || name.equals("prod") || name.equals("mean")
					|| name.equals("variance") || name.equals("sd") || name.equals("dot_self")
					|| name.equals("distance") || name.equals("squared_distance")
					|| name.equals("dot_product") || name.equals("num_elements") || name.equals("size")
					|| name.equals("rows") || name.equals("cols") || name.startsWith("rep_")
					|| name.equals("to_vector") || name.equals("to_row_vector")
					|| name.equals("to_array_1d") || name.equals("transpose")
					|| name.equals("identity_matrix") || name.equals("diag_matrix")
					|| name.equals("diagonal") || name.equals("cholesky_decompose")
					|| name.equals("qr_thin_Q") || name.equals("qr_thin_R")
					|| name.equals("inverse") || name.equals("determinant")
					|| name.equals("log_determinant") || name.equals("log_determinant_spd")
					|| name.equals("mdivide_left_spd") || name.equals("mdivide_right_spd")
					|| name.equals("inverse_spd") || name.equals("multiply_lower_tri_self_transpose")
					|| name.equals("symmetrize_from_lower_tri") || name.equals("symmetrize_from_upper_tri")
					|| name.equals("append_array") || name.equals("append_row") || name.equals("append_col")
					|| name.equals("head") || name.equals("tail") || name.equals("segment")
					|| name.equals("block") || name.equals("row") || name.equals("col")
					|| name.equals("trace") || name.equals("quad_form") || name.equals("trace_quad_form")
					|| name.equals("trace_gen_quad_form")
					|| name.equals("diag_pre_multiply") || name.equals("diag_post_multiply")
					|| name.equals("crossprod") || name.equals("tcrossprod")
					|| name.equals("columns_dot_product") || name.equals("rows_dot_product")
					|| name.equals("columns_dot_self") || name.equals("rows_dot_self")
					|| name.equals("softmax") || name.equals("log_softmax")
					|| name.equals("cumulative_sum") || name.equals("reverse")
					|| name.equals("sort_asc") || name.equals("sort_desc")
					|| name.equals("csr_matrix_times_vector") || name.equals("csr_to_dense_matrix")
					|| name.equals("csr_extract_w") || name.equals("csr_extract_v")
					|| name.equals("csr_extract_u") || name.equals("algebra_solver")
					|| name.equals("algebra_solver_newton") || name.equals("ode_rk45")
					|| name.equals("ode_bdf") || name.equals("dae") || name.equals("to_complex")
					|| name.equals("get_real") || name.equals("get_imag") || name.equals("conj")
					|| name.equals("arg") || name.equals("norm") || name.equals("integrate_1d")
					|| arguments.size() == 1 && (name.equals("min") || name.equals("max")
							|| name.equals("log_sum_exp"))) return null;
			if (name.endsWith("_rng") || name.endsWith("_lpdf") || name.endsWith("_lpmf")
					|| name.endsWith("_lupdf") || name.endsWith("_lupmf")) return null;
			RuntimeValue[] evaluated = new RuntimeValue[arguments.size()]; RuntimeValue template = null;
			for (int i = 0; i < evaluated.length; i++) {
				evaluated[i] = arguments.get(i).evalValue(context);
				if (!evaluated[i].isScalar()) {
					if (template != null && (!Arrays.equals(template.shape, evaluated[i].shape)
							|| template.kind != evaluated[i].kind || template.arrayRank != evaluated[i].arrayRank))
						throw new IllegalArgumentException(name + " cannot broadcast " + template.kind
								+ Arrays.toString(template.shape) + " with " + evaluated[i].kind
								+ Arrays.toString(evaluated[i].shape));
					template = evaluated[i];
				}
			}
			int length = template == null ? 1 : template.values.length; Diff[] result = new Diff[length];
			for (int element = 0; element < length; element++) {
				Diff[] scalar = new Diff[evaluated.length];
				for (int argument = 0; argument < evaluated.length; argument++)
					scalar[argument] = evaluated[argument].values[evaluated[argument].isScalar() ? 0 : element];
				result[element] = scalarFunction(name, scalar, context);
				if (result[element] == null) return null;
			}
			return template == null ? RuntimeValue.scalar(result[0])
					: new RuntimeValue(result, template.shape, template.kind, template.arrayRank);
		}
		private RuntimeValue structuredFunction(Context context) {
			if (name.equals("to_complex") && (arguments.size()==1||arguments.size()==2)) {
				RuntimeValue real=arguments.get(0).evalValue(context);
				if(arguments.size()==1) return promoteComplex(real,complexKindFor(real.kind),context);
				RuntimeValue imaginary=arguments.get(1).evalValue(context);
				if(real.kind!=imaginary.kind||real.arrayRank!=imaginary.arrayRank||!Arrays.equals(real.shape,imaginary.shape))
					throw new IllegalArgumentException("to_complex arguments must have identical real shapes");
				Diff[] values=new Diff[real.values.length*2]; for(int i=0;i<real.values.length;i++){values[2*i]=real.values[i];values[2*i+1]=imaginary.values[i];}
				return new RuntimeValue(values,real.shape,complexKindFor(real.kind),real.arrayRank);
			}
			if ((name.equals("get_real")||name.equals("get_imag")||name.equals("conj")
					||name.equals("abs")||name.equals("arg")||name.equals("norm")
					||name.equals("exp")||name.equals("log")||name.equals("sqrt")
					||name.equals("sin")||name.equals("cos")||name.equals("tan")
					||name.equals("sinh")||name.equals("cosh")||name.equals("tanh"))&&arguments.size()==1) {
				RuntimeValue input=arguments.get(0).evalValue(context);
				if(input.isComplex()) return complexFunction(input,context);
			}
			if(name.equals("sum")&&arguments.size()==1) {
				RuntimeValue input=arguments.get(0).evalValue(context);
				if(input.isComplex()) {
					Diff real=Diff.constant(0,context.dimension),imaginary=Diff.constant(0,context.dimension);
					for(int i=0;i<input.logicalSize();i++){real=real.add(input.values[2*i]);imaginary=imaginary.add(input.values[2*i+1]);}
					return RuntimeValue.complex(real,imaginary);
				}
			}
			if ((name.equals("min") || name.equals("max") || name.equals("log_sum_exp")
					|| name.equals("mean") || name.equals("variance") || name.equals("sd")
					|| name.equals("dot_self")) && arguments.size() == 1)
				return RuntimeValue.scalar(reduction(arguments.get(0).evalValue(context), context));
			if ((name.equals("distance") || name.equals("squared_distance")) && arguments.size() == 2)
				return RuntimeValue.scalar(distance(arguments.get(0).evalValue(context),
						arguments.get(1).evalValue(context), context));
			if ((name.equals("algebra_solver") || name.equals("algebra_solver_newton"))
					&& arguments.size() == 5) return algebraSolver(context);
			if (name.equals("integrate_1d") && (arguments.size() == 6 || arguments.size() == 7))
				return integrateOneDimension(context);
			if ((name.equals("ode_rk45") || name.equals("ode_bdf"))
					&& arguments.size() == 7) return odeSolver(context, name.equals("ode_bdf"));
			if (name.equals("dae") && arguments.size() == 8) return daeSolver(context);
			if (name.equals("transpose") && arguments.size() == 1)
				return new TransposeExpr(arguments.get(0)).evalValue(context);
			if (name.equals("rep_matrix") && arguments.size() == 3) {
				Diff value = arguments.get(0).eval(context);
				int rows = checkedPositiveSize(arguments.get(1).eval(context).value, name);
				int columns = checkedPositiveSize(arguments.get(2).eval(context).value, name);
				Diff[] result = new Diff[rows * columns]; Arrays.fill(result, value);
				return new RuntimeValue(result, new int[] {rows, columns}, ValueKind.MATRIX, 0);
			}
			if (name.equals("rep_matrix") && arguments.size() == 2) {
				RuntimeValue value = arguments.get(0).evalValue(context); requireVector(value, name);
				int repetitions = checkedPositiveSize(arguments.get(1).eval(context).value, name);
				int rows = value.kind == ValueKind.VECTOR ? value.values.length : repetitions;
				int columns = value.kind == ValueKind.VECTOR ? repetitions : value.values.length;
				Diff[] result = new Diff[rows * columns];
				for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
					result[row * columns + column] = value.values[value.kind == ValueKind.VECTOR ? row : column];
				return new RuntimeValue(result, new int[] {rows, columns}, ValueKind.MATRIX, 0);
			}
			if (name.equals("identity_matrix") && arguments.size() == 1) {
				int size = checkedPositiveSize(arguments.get(0).eval(context).value, name);
				Diff[] result = constants(size * size, 0.0, context);
				for (int i = 0; i < size; i++) result[i * size + i] = Diff.constant(1.0, context.dimension);
				return new RuntimeValue(result, new int[] {size, size}, ValueKind.MATRIX, 0);
			}
			if (name.equals("diag_matrix") && arguments.size() == 1) {
				RuntimeValue vector = arguments.get(0).evalValue(context);
				requireVector(vector, name); int size = vector.values.length;
				Diff[] result = constants(size * size, 0.0, context);
				for (int i = 0; i < size; i++) result[i * size + i] = vector.values[i];
				return new RuntimeValue(result, new int[] {size, size}, ValueKind.MATRIX, 0);
			}
			if (name.equals("diagonal") && arguments.size() == 1) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireMatrix(matrix, name);
				int size = Math.min(matrix.rows(), matrix.columns()); Diff[] result = new Diff[size];
				for (int i = 0; i < size; i++) result[i] = matrix.values[i * matrix.columns() + i];
				return new RuntimeValue(result, new int[] {size}, ValueKind.VECTOR, 0);
			}
			if (name.equals("cholesky_decompose") && arguments.size() == 1) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireSquareMatrix(matrix, name);
				return cholesky(matrix, context);
			}
			if ((name.equals("qr_thin_Q") || name.equals("qr_thin_R")) && arguments.size() == 1) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireMatrix(matrix, name);
				QrPair qr = qrThin(matrix, context);
				return name.equals("qr_thin_Q") ? qr.q : qr.r;
			}
			if ((name.equals("inverse") || name.equals("inverse_spd")) && arguments.size() == 1) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireSquareMatrix(matrix, name);
				if (name.equals("inverse_spd")) cholesky(matrix, context);
				return inverse(matrix, context);
			}
			if ((name.equals("determinant") || name.equals("log_determinant")
					|| name.equals("log_determinant_spd")) && arguments.size() == 1) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireSquareMatrix(matrix, name);
				if (name.equals("log_determinant_spd")) cholesky(matrix, context);
				Diff determinant = determinant(matrix, context);
				return RuntimeValue.scalar(name.startsWith("log_determinant") ? determinant.abs().log() : determinant);
			}
			if (name.equals("mdivide_left_spd") && arguments.size() == 2) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireSquareMatrix(matrix, name);
				RuntimeValue right = arguments.get(1).evalValue(context);
				return solveSpd(matrix, right, context);
			}
			if (name.equals("mdivide_right_spd") && arguments.size() == 2) {
				RuntimeValue left = arguments.get(0).evalValue(context);
				RuntimeValue matrix = arguments.get(1).evalValue(context); requireSquareMatrix(matrix, name);
				RuntimeValue transposed = transposeValue(left, context);
				RuntimeValue solved = solveSpd(matrix, transposed, context);
				return transposeValue(solved, context);
			}
			if (name.equals("multiply_lower_tri_self_transpose") && arguments.size() == 1) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireMatrix(matrix, name);
				Diff[] lower = constants(matrix.values.length, 0.0, context);
				for (int row = 0; row < matrix.rows(); row++)
					for (int column = 0; column < matrix.columns() && column <= row; column++)
						lower[row * matrix.columns() + column] = matrix.values[row * matrix.columns() + column];
				RuntimeValue triangular = new RuntimeValue(lower, matrix.shape, ValueKind.MATRIX, 0);
				return matrixProduct(triangular, transposeValue(triangular, context), context);
			}
			if ((name.equals("symmetrize_from_lower_tri") || name.equals("symmetrize_from_upper_tri"))
					&& arguments.size() == 1)
				return symmetrize(arguments.get(0).evalValue(context), name.endsWith("lower_tri"), context);
			if (name.equals("append_array") && arguments.size() == 2)
				return appendArray(arguments.get(0).evalValue(context), arguments.get(1).evalValue(context));
			if ((name.equals("append_row") || name.equals("append_col")) && arguments.size() == 2)
				return appendContainer(arguments.get(0).evalValue(context),
						arguments.get(1).evalValue(context), name.equals("append_row"));
			if ((name.equals("head") || name.equals("tail")) && arguments.size() == 2) {
				RuntimeValue value = arguments.get(0).evalValue(context);
				int length = checkedPositiveSize(arguments.get(1).eval(context).value, name);
				return sliceFirst(value, name.equals("head") ? 0 : firstExtent(value) - length, length, name);
			}
			if (name.equals("segment") && arguments.size() == 3) {
				RuntimeValue value = arguments.get(0).evalValue(context);
				int start = checkedPositiveSize(arguments.get(1).eval(context).value, name) - 1;
				int length = checkedPositiveSize(arguments.get(2).eval(context).value, name);
				return sliceFirst(value, start, length, name);
			}
			if ((name.equals("row") || name.equals("col")) && arguments.size() == 2) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireMatrix(matrix, name);
				int index = checkedPositiveSize(arguments.get(1).eval(context).value, name) - 1;
				return name.equals("row") ? matrixRow(matrix, index) : matrixColumn(matrix, index);
			}
			if (name.equals("block") && arguments.size() == 5) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireMatrix(matrix, name);
				int row = checkedPositiveSize(arguments.get(1).eval(context).value, name) - 1;
				int column = checkedPositiveSize(arguments.get(2).eval(context).value, name) - 1;
				int rows = checkedPositiveSize(arguments.get(3).eval(context).value, name);
				int columns = checkedPositiveSize(arguments.get(4).eval(context).value, name);
				return matrixBlock(matrix, row, column, rows, columns);
			}
			if (name.equals("trace") && arguments.size() == 1) {
				RuntimeValue matrix = arguments.get(0).evalValue(context); requireMatrix(matrix, name);
				Diff result = Diff.constant(0.0, context.dimension);
				for (int i = 0; i < Math.min(matrix.rows(), matrix.columns()); i++)
					result = result.add(matrix.values[i * matrix.columns() + i]);
				return RuntimeValue.scalar(result);
			}
			if (name.equals("quad_form") && arguments.size() == 2) {
				Expr matrix = arguments.get(0), right = arguments.get(1);
				return new BinaryExpr("*", new BinaryExpr("*", new TransposeExpr(right), matrix), right)
						.evalValue(context);
			}
			if (name.equals("trace_quad_form") && arguments.size() == 2) {
				RuntimeValue matrix = arguments.get(0).evalValue(context);
				RuntimeValue right = arguments.get(1).evalValue(context);
				RuntimeValue product = matrixProduct(transposeValue(right, context),
						matrixProduct(matrix, right, context), context);
				return RuntimeValue.scalar(trace(product, context));
			}
			if (name.equals("trace_gen_quad_form") && arguments.size() == 3) {
				RuntimeValue diagonal = arguments.get(0).evalValue(context);
				RuntimeValue matrix = arguments.get(1).evalValue(context);
				RuntimeValue right = arguments.get(2).evalValue(context);
				RuntimeValue quadratic = matrixProduct(transposeValue(right, context),
						matrixProduct(matrix, right, context), context);
				return RuntimeValue.scalar(trace(matrixProduct(diagonal, quadratic, context), context));
			}
			if ((name.equals("diag_pre_multiply") || name.equals("diag_post_multiply"))
					&& arguments.size() == 2)
				return diagonalMultiply(arguments.get(0).evalValue(context),
						arguments.get(1).evalValue(context), name.equals("diag_pre_multiply"));
			if ((name.equals("crossprod") || name.equals("tcrossprod")) && arguments.size() == 1) {
				Expr matrix = arguments.get(0);
				return name.equals("crossprod")
						? new BinaryExpr("*", new TransposeExpr(matrix), matrix).evalValue(context)
						: new BinaryExpr("*", matrix, new TransposeExpr(matrix)).evalValue(context);
			}
			if ((name.equals("columns_dot_product") || name.equals("rows_dot_product"))
					&& arguments.size() == 2)
				return matrixDotProducts(arguments.get(0).evalValue(context),
						arguments.get(1).evalValue(context), name.equals("columns_dot_product"), context);
			if ((name.equals("columns_dot_self") || name.equals("rows_dot_self"))
					&& arguments.size() == 1) {
				RuntimeValue matrix = arguments.get(0).evalValue(context);
				return matrixDotProducts(matrix, matrix, name.equals("columns_dot_self"), context);
			}
			if ((name.equals("softmax") || name.equals("log_softmax")
					|| name.equals("cumulative_sum") || name.equals("reverse")
					|| name.equals("sort_asc") || name.equals("sort_desc")) && arguments.size() == 1)
				return sequenceTransform(arguments.get(0).evalValue(context), context);
			if (name.equals("csr_matrix_times_vector") && arguments.size() == 6)
				return csrMultiply(context);
			if (name.equals("csr_to_dense_matrix") && arguments.size() == 5)
				return csrDense(context);
			if ((name.equals("csr_extract_w") || name.equals("csr_extract_v")
					|| name.equals("csr_extract_u")) && arguments.size() == 1)
				return csrExtract(arguments.get(0).evalValue(context), context);
			if ((name.equals("multi_normal_lpdf") || name.equals("multi_normal_cholesky_lpdf"))
					&& arguments.size() == 3)
				return RuntimeValue.scalar(multiNormal(context, name.endsWith("cholesky_lpdf")));
			return null;
		}
		private RuntimeValue appendArray(RuntimeValue left, RuntimeValue right) {
			if (left.arrayRank < 1 || right.arrayRank != left.arrayRank || left.kind != right.kind
					|| left.shape.length != right.shape.length)
				throw new IllegalArgumentException("append_array requires arrays with a common element type");
			for (int i = 1; i < left.shape.length; i++) if (left.shape[i] != right.shape[i])
				throw new IllegalArgumentException("append_array element shapes differ");
			Diff[] values = new Diff[left.values.length + right.values.length];
			System.arraycopy(left.values, 0, values, 0, left.values.length);
			System.arraycopy(right.values, 0, values, left.values.length, right.values.length);
			int[] shape = left.shape.clone(); shape[0] += right.shape[0];
			return new RuntimeValue(values, shape, left.kind, left.arrayRank);
		}
		private RuntimeValue appendContainer(RuntimeValue left, RuntimeValue right, boolean byRows) {
			if (left.arrayRank != 0 || right.arrayRank != 0)
				throw new IllegalArgumentException(name + " does not accept arrays; use append_array");
			if (left.kind == ValueKind.MATRIX && right.kind == ValueKind.MATRIX) {
				if (byRows && left.columns() != right.columns()
						|| !byRows && left.rows() != right.rows())
					throw new IllegalArgumentException(name + " matrix dimension mismatch");
				int rows = byRows ? left.rows() + right.rows() : left.rows();
				int columns = byRows ? left.columns() : left.columns() + right.columns();
				Diff[] values = new Diff[rows * columns];
				if (byRows) {
					System.arraycopy(left.values, 0, values, 0, left.values.length);
					System.arraycopy(right.values, 0, values, left.values.length, right.values.length);
				} else for (int row = 0; row < rows; row++) {
					System.arraycopy(left.values, row * left.columns(), values, row * columns, left.columns());
					System.arraycopy(right.values, row * right.columns(), values,
							row * columns + left.columns(), right.columns());
				}
				return new RuntimeValue(values, new int[] {rows, columns}, ValueKind.MATRIX, 0);
			}
			if (left.kind != right.kind || left.shape.length != 1 || right.shape.length != 1)
				throw new IllegalArgumentException(name + " requires compatible vectors or matrices");
			if (byRows && left.kind == ValueKind.VECTOR || !byRows && left.kind == ValueKind.ROW_VECTOR) {
				Diff[] values = new Diff[left.values.length + right.values.length];
				System.arraycopy(left.values, 0, values, 0, left.values.length);
				System.arraycopy(right.values, 0, values, left.values.length, right.values.length);
				return new RuntimeValue(values, new int[] {values.length}, left.kind, 0);
			}
			if (left.values.length != right.values.length)
				throw new IllegalArgumentException(name + " vector dimension mismatch");
			int rows = byRows ? 2 : left.values.length, columns = byRows ? left.values.length : 2;
			Diff[] values = new Diff[rows * columns];
			if (byRows) {
				System.arraycopy(left.values, 0, values, 0, columns);
				System.arraycopy(right.values, 0, values, columns, columns);
			} else for (int row = 0; row < rows; row++) {
				values[row * 2] = left.values[row]; values[row * 2 + 1] = right.values[row];
			}
			return new RuntimeValue(values, new int[] {rows, columns}, ValueKind.MATRIX, 0);
		}
		private RuntimeValue matrixDotProducts(RuntimeValue left, RuntimeValue right,
				boolean columns, Context context) {
			requireMatrix(left, name); requireMatrix(right, name);
			if (!Arrays.equals(left.shape, right.shape))
				throw new IllegalArgumentException(name + " matrix dimension mismatch");
			int count = columns ? left.columns() : left.rows();
			int terms = columns ? left.rows() : left.columns();
			Diff[] result = constants(count, 0.0, context);
			for (int group = 0; group < count; group++) for (int term = 0; term < terms; term++) {
				int index = columns ? term * left.columns() + group : group * left.columns() + term;
				result[group] = result[group].add(left.values[index].multiply(right.values[index]));
			}
			return new RuntimeValue(result, new int[] {count},
					columns ? ValueKind.ROW_VECTOR : ValueKind.VECTOR, 0);
		}
		private Diff reduction(RuntimeValue input, Context context) {
			if (input.kind == ValueKind.TUPLE || input.isComplex())
				throw new IllegalArgumentException(name + " requires a real scalar or container");
			if (input.values.length == 0) throw new IllegalArgumentException(name + " requires a non-empty input");
			if (name.equals("min") || name.equals("max")) {
				Diff result = input.values[0];
				for (int i = 1; i < input.values.length; i++)
					if (name.equals("min") ? input.values[i].value < result.value
							: input.values[i].value > result.value) result = input.values[i];
				return result;
			}
			if (name.equals("log_sum_exp")) {
				double maximum = Double.NEGATIVE_INFINITY;
				for (Diff value : input.values) maximum = Math.max(maximum, value.value);
				Diff sum = Diff.constant(0.0, context.dimension);
				for (Diff value : input.values) sum = sum.add(value.add(-maximum).exp());
				return sum.log().add(maximum);
			}
			if (name.equals("dot_self")) {
				Diff[] partials = input.values; double[] derivative = new double[partials.length]; double value = 0;
				for (int i = 0; i < partials.length; i++) {
					value += partials[i].value * partials[i].value; derivative[i] = 2 * partials[i].value;
				}
				return Diff.atomic(value, partials, derivative);
			}
			Diff mean = Diff.constant(0.0, context.dimension);
			for (Diff value : input.values) mean = mean.add(value);
			mean = mean.multiply(1.0 / input.values.length);
			if (name.equals("mean")) return mean;
			if (input.values.length < 2) throw new IllegalArgumentException(name + " requires at least two values");
			Diff sumSquares = Diff.constant(0.0, context.dimension);
			for (Diff value : input.values) {
				Diff centered = value.subtract(mean); sumSquares = sumSquares.add(centered.multiply(centered));
			}
			Diff variance = sumSquares.multiply(1.0 / (input.values.length - 1.0));
			return name.equals("sd") ? variance.sqrt() : variance;
		}
		private Diff distance(RuntimeValue left, RuntimeValue right, Context context) {
			if (left.kind == ValueKind.TUPLE || right.kind == ValueKind.TUPLE)
				throw new IllegalArgumentException(name + " does not accept tuples");
			if (left.values.length != right.values.length || !Arrays.equals(left.shape, right.shape)
					|| left.kind != right.kind || left.arrayRank != right.arrayRank)
				throw new IllegalArgumentException(name + " requires identically typed and shaped values");
			Diff[] inputs = new Diff[left.values.length * 2]; double[] derivative = new double[inputs.length];
			double squared = 0;
			for (int i = 0; i < left.values.length; i++) {
				double difference = left.values[i].value - right.values[i].value; squared += difference * difference;
				inputs[i] = left.values[i]; inputs[left.values.length + i] = right.values[i];
				derivative[i] = 2 * difference; derivative[left.values.length + i] = -2 * difference;
			}
			Diff result = Diff.atomic(squared, inputs, derivative);
			return name.equals("distance") ? result.sqrt() : result;
		}
		private RuntimeValue transposeValue(RuntimeValue value, Context context) {
			if (value.kind == ValueKind.VECTOR)
				return new RuntimeValue(value.values, value.shape, ValueKind.ROW_VECTOR, 0);
			if (value.kind == ValueKind.ROW_VECTOR)
				return new RuntimeValue(value.values, value.shape, ValueKind.VECTOR, 0);
			requireMatrix(value, name); int rows = value.rows(), columns = value.columns();
			Diff[] result = new Diff[value.values.length];
			for (int row = 0; row < rows; row++) for (int column = 0; column < columns; column++)
				result[column * rows + row] = value.values[row * columns + column];
			return new RuntimeValue(result, new int[] {columns, rows}, ValueKind.MATRIX, 0);
		}
		private RuntimeValue matrixProduct(RuntimeValue left, RuntimeValue right, Context context) {
			if (left.kind == ValueKind.MATRIX && right.kind == ValueKind.MATRIX) {
				if (left.columns() != right.rows()) throw new IllegalArgumentException(name + " matrix dimension mismatch");
				Diff[] result = constants(left.rows() * right.columns(), 0.0, context);
				for (int row = 0; row < left.rows(); row++) for (int column = 0; column < right.columns(); column++)
					for (int k = 0; k < left.columns(); k++) result[row * right.columns() + column] =
							result[row * right.columns() + column].add(left.values[row * left.columns() + k]
									.multiply(right.values[k * right.columns() + column]));
				return new RuntimeValue(result, new int[] {left.rows(), right.columns()}, ValueKind.MATRIX, 0);
			}
			throw new IllegalArgumentException(name + " requires matrix operands");
		}
		private Diff trace(RuntimeValue matrix, Context context) {
			requireMatrix(matrix, name); Diff result = Diff.constant(0.0, context.dimension);
			for (int i = 0; i < Math.min(matrix.rows(), matrix.columns()); i++)
				result = result.add(matrix.values[i * matrix.columns() + i]);
			return result;
		}
		private RuntimeValue symmetrize(RuntimeValue matrix, boolean lower, Context context) {
			requireSquareMatrix(matrix, name); int size = matrix.rows(); Diff[] result = constants(size * size, 0, context);
			for (int row = 0; row < size; row++) for (int column = 0; column < size; column++)
				result[row * size + column] = lower
						? matrix.values[(row >= column ? row * size + column : column * size + row)]
						: matrix.values[(row <= column ? row * size + column : column * size + row)];
			return new RuntimeValue(result, matrix.shape, ValueKind.MATRIX, 0);
		}
		private int firstExtent(RuntimeValue value) {
			if (value.shape.length == 0) throw new IllegalArgumentException(name + " requires a sequence");
			if (value.arrayRank == 0 && value.kind == ValueKind.MATRIX)
				throw new IllegalArgumentException(name + " requires an array, vector, or row_vector");
			return value.shape[0];
		}
		private RuntimeValue sliceFirst(RuntimeValue value, int start, int length, String function) {
			int extent = firstExtent(value);
			if (start < 0 || length < 0 || start + length > extent)
				throw new IllegalArgumentException(function + " range exceeds sequence length " + extent);
			int block = value.values.length / extent;
			Diff[] result = new Diff[length * block];
			System.arraycopy(value.values, start * block, result, 0, result.length);
			int[] shape = value.shape.clone(); shape[0] = length;
			return new RuntimeValue(result, shape, value.kind, value.arrayRank);
		}
		private RuntimeValue matrixRow(RuntimeValue matrix, int row) {
			if (row < 0 || row >= matrix.rows()) throw new IllegalArgumentException("row index is out of range");
			Diff[] result = new Diff[matrix.columns()];
			System.arraycopy(matrix.values, row * matrix.columns(), result, 0, result.length);
			return new RuntimeValue(result, new int[] {result.length}, ValueKind.ROW_VECTOR, 0);
		}
		private RuntimeValue matrixColumn(RuntimeValue matrix, int column) {
			if (column < 0 || column >= matrix.columns()) throw new IllegalArgumentException("column index is out of range");
			Diff[] result = new Diff[matrix.rows()];
			for (int row = 0; row < result.length; row++) result[row] = matrix.values[row * matrix.columns() + column];
			return new RuntimeValue(result, new int[] {result.length}, ValueKind.VECTOR, 0);
		}
		private RuntimeValue matrixBlock(RuntimeValue matrix, int firstRow, int firstColumn,
				int rows, int columns) {
			if (firstRow < 0 || firstColumn < 0 || firstRow + rows > matrix.rows()
					|| firstColumn + columns > matrix.columns())
				throw new IllegalArgumentException("block exceeds matrix dimensions");
			Diff[] result = new Diff[rows * columns];
			for (int row = 0; row < rows; row++)
				System.arraycopy(matrix.values, (firstRow + row) * matrix.columns() + firstColumn,
						result, row * columns, columns);
			return new RuntimeValue(result, new int[] {rows, columns}, ValueKind.MATRIX, 0);
		}
		private RuntimeValue diagonalMultiply(RuntimeValue diagonal, RuntimeValue matrix, boolean before) {
			requireVector(diagonal, name); requireMatrix(matrix, name);
			int expected = before ? matrix.rows() : matrix.columns();
			if (diagonal.values.length != expected) throw new IllegalArgumentException(name + " dimension mismatch");
			Diff[] result = new Diff[matrix.values.length];
			for (int row = 0; row < matrix.rows(); row++) for (int column = 0; column < matrix.columns(); column++)
				result[row * matrix.columns() + column] = matrix.values[row * matrix.columns() + column]
						.multiply(diagonal.values[before ? row : column]);
			return new RuntimeValue(result, matrix.shape, ValueKind.MATRIX, 0);
		}
		private RuntimeValue sequenceTransform(RuntimeValue value, Context context) {
			requireVector(value, name); Diff[] result = value.values.clone();
			if (name.equals("reverse")) {
				for (int i = 0; i < result.length / 2; i++) { Diff swap = result[i]; result[i] = result[result.length - 1 - i]; result[result.length - 1 - i] = swap; }
			} else if (name.equals("sort_asc") || name.equals("sort_desc")) {
				final boolean ascending = name.equals("sort_asc");
				Arrays.sort(result, (a, b) -> ascending ? Double.compare(a.value, b.value) : Double.compare(b.value, a.value));
			} else if (name.equals("cumulative_sum")) {
				for (int i = 1; i < result.length; i++) result[i] = result[i - 1].add(result[i]);
			} else {
				double maximum = Double.NEGATIVE_INFINITY;
				for (Diff element : result) maximum = Math.max(maximum, element.value);
				Diff denominator = Diff.constant(0.0, context.dimension);
				for (int i = 0; i < result.length; i++) { result[i] = result[i].add(-maximum).exp(); denominator = denominator.add(result[i]); }
				for (int i = 0; i < result.length; i++) result[i] = name.equals("softmax")
						? result[i].divide(denominator) : result[i].log().subtract(denominator.log());
			}
			return new RuntimeValue(result, value.shape, value.kind, 0);
		}
		private RuntimeValue complexFunction(RuntimeValue input,Context context) {
			boolean realOutput=name.equals("get_real")||name.equals("get_imag")||name.equals("abs")
					||name.equals("arg")||name.equals("norm");
			Diff[] result=new Diff[input.logicalSize()*(realOutput?1:2)];
			for(int i=0;i<input.logicalSize();i++) {
				Diff real=input.values[2*i],imaginary=input.values[2*i+1];
				if(name.equals("get_real")) result[i]=real;
				else if(name.equals("get_imag")) result[i]=imaginary;
				else if(name.equals("abs")) result[i]=real.multiply(real).add(imaginary.multiply(imaginary)).sqrt();
				else if(name.equals("norm")) result[i]=real.multiply(real).add(imaginary.multiply(imaginary));
				else if(name.equals("arg")) {
					double denominator=real.value*real.value+imaginary.value*imaginary.value;
					result[i]=Diff.atomic(Math.atan2(imaginary.value,real.value),new Diff[]{real,imaginary},
							new double[]{-imaginary.value/denominator,real.value/denominator});
				} else if(name.equals("conj")) {result[2*i]=real;result[2*i+1]=imaginary.negate();}
				else if(name.equals("exp")) {Diff scale=real.exp();result[2*i]=scale.multiply(imaginary.cos());result[2*i+1]=scale.multiply(imaginary.sin());}
				else if(name.equals("log")) {result[2*i]=real.multiply(real).add(imaginary.multiply(imaginary)).log().multiply(.5);result[2*i+1]=Diff.atomic(Math.atan2(imaginary.value,real.value),new Diff[]{real,imaginary},new double[]{-imaginary.value/(real.value*real.value+imaginary.value*imaginary.value),real.value/(real.value*real.value+imaginary.value*imaginary.value)});}
				else if(name.equals("sin")){result[2*i]=real.sin().multiply(imaginary.cosh());result[2*i+1]=real.cos().multiply(imaginary.sinh());}
				else if(name.equals("cos")){result[2*i]=real.cos().multiply(imaginary.cosh());result[2*i+1]=real.sin().multiply(imaginary.sinh()).negate();}
				else if(name.equals("sinh")){result[2*i]=real.sinh().multiply(imaginary.cos());result[2*i+1]=real.cosh().multiply(imaginary.sin());}
				else if(name.equals("cosh")){result[2*i]=real.cosh().multiply(imaginary.cos());result[2*i+1]=real.sinh().multiply(imaginary.sin());}
				else if(name.equals("tan")||name.equals("tanh")){
					Diff numeratorReal,numeratorImaginary,denominatorReal,denominatorImaginary;
					if(name.equals("tan")){numeratorReal=real.sin().multiply(imaginary.cosh());numeratorImaginary=real.cos().multiply(imaginary.sinh());denominatorReal=real.cos().multiply(imaginary.cosh());denominatorImaginary=real.sin().multiply(imaginary.sinh()).negate();}
					else{numeratorReal=real.sinh().multiply(imaginary.cos());numeratorImaginary=real.cosh().multiply(imaginary.sin());denominatorReal=real.cosh().multiply(imaginary.cos());denominatorImaginary=real.sinh().multiply(imaginary.sin());}
					Diff denominator=denominatorReal.multiply(denominatorReal).add(denominatorImaginary.multiply(denominatorImaginary));
					result[2*i]=numeratorReal.multiply(denominatorReal).add(numeratorImaginary.multiply(denominatorImaginary)).divide(denominator);
					result[2*i+1]=numeratorImaginary.multiply(denominatorReal).subtract(numeratorReal.multiply(denominatorImaginary)).divide(denominator);
				}
				else { double magnitude=Math.hypot(real.value,imaginary.value),u=Math.sqrt((magnitude+real.value)/2),v=Math.copySign(Math.sqrt(Math.max(0,(magnitude-real.value)/2)),imaginary.value); double denominator=2*(u*u+v*v); result[2*i]=Diff.atomic(u,new Diff[]{real,imaginary},new double[]{u/denominator,v/denominator});result[2*i+1]=Diff.atomic(v,new Diff[]{real,imaginary},new double[]{-v/denominator,u/denominator}); }
			}
			return new RuntimeValue(result,input.shape,realOutput?realKindFor(input.kind):input.kind,input.arrayRank);
		}
		private RuntimeValue csrMultiply(Context context) {
			int rows = checkedPositiveSize(arguments.get(0).eval(context).value, name);
			int columns = checkedPositiveSize(arguments.get(1).eval(context).value, name);
			RuntimeValue weights = arguments.get(2).evalValue(context);
			RuntimeValue columnIndex = arguments.get(3).evalValue(context);
			RuntimeValue rowStart = arguments.get(4).evalValue(context);
			RuntimeValue vector = arguments.get(5).evalValue(context); requireVector(vector, name);
			int[] v = integerArray(columnIndex, name), u = integerArray(rowStart, name);
			validateCsr(rows, columns, weights.values.length, v, u);
			if (vector.values.length != columns) throw new IllegalArgumentException(name+" vector length mismatch");
			Diff[] result = constants(rows, 0.0, context);
			for (int row = 0; row < rows; row++) for (int offset = u[row]-1; offset < u[row+1]-1; offset++)
				result[row] = result[row].add(weights.values[offset].multiply(vector.values[v[offset]-1]));
			return new RuntimeValue(result, new int[] {rows}, ValueKind.VECTOR, 0);
		}
		private RuntimeValue csrDense(Context context) {
			int rows = checkedPositiveSize(arguments.get(0).eval(context).value, name);
			int columns = checkedPositiveSize(arguments.get(1).eval(context).value, name);
			RuntimeValue weights = arguments.get(2).evalValue(context);
			int[] v = integerArray(arguments.get(3).evalValue(context), name);
			int[] u = integerArray(arguments.get(4).evalValue(context), name);
			validateCsr(rows, columns, weights.values.length, v, u);
			Diff[] result = constants(rows*columns, 0.0, context);
			for (int row = 0; row < rows; row++) for (int offset = u[row]-1; offset < u[row+1]-1; offset++)
				result[row*columns+v[offset]-1] = result[row*columns+v[offset]-1].add(weights.values[offset]);
			return new RuntimeValue(result, new int[] {rows, columns}, ValueKind.MATRIX, 0);
		}
		private RuntimeValue csrExtract(RuntimeValue matrix, Context context) {
			requireMatrix(matrix, name); int nonzero = 0;
			for (Diff value : matrix.values) if (value.value != 0.0) nonzero++;
			if (name.equals("csr_extract_w")) {
				Diff[] result = new Diff[nonzero]; int output = 0;
				for (Diff value : matrix.values) if (value.value != 0.0) result[output++] = value;
				return new RuntimeValue(result, new int[] {nonzero}, ValueKind.VECTOR, 0);
			}
			if (name.equals("csr_extract_v")) {
				Diff[] result = new Diff[nonzero]; int output = 0;
				for (int row = 0; row < matrix.rows(); row++) for (int column = 0; column < matrix.columns(); column++)
					if (matrix.values[row*matrix.columns()+column].value != 0.0)
						result[output++] = Diff.constant(column+1, context.dimension);
				return new RuntimeValue(result, new int[] {nonzero}, ValueKind.SCALAR, 1);
			}
			Diff[] result = new Diff[matrix.rows()+1]; int offset = 1; result[0] = Diff.constant(1, context.dimension);
			for (int row = 0; row < matrix.rows(); row++) {
				for (int column = 0; column < matrix.columns(); column++)
					if (matrix.values[row*matrix.columns()+column].value != 0.0) offset++;
				result[row+1] = Diff.constant(offset, context.dimension);
			}
			return new RuntimeValue(result, new int[] {result.length}, ValueKind.SCALAR, 1);
		}
		private int[] integerArray(RuntimeValue value, String function) {
			int[] result = new int[value.values.length];
			for (int i = 0; i < result.length; i++) {
				double element = value.values[i].value;
				if (element != Math.rint(element) || element < Integer.MIN_VALUE || element > Integer.MAX_VALUE)
					throw new IllegalArgumentException(function+" CSR indexes must be integers");
				result[i] = (int)element;
			}
			return result;
		}
		private void validateCsr(int rows, int columns, int nonzero, int[] v, int[] u) {
			if (v.length != nonzero || u.length != rows+1 || u[0] != 1 || u[rows] != nonzero+1)
				throw new IllegalArgumentException(name+" invalid CSR storage lengths or row endpoints");
			for (int row = 0; row < rows; row++) if (u[row] > u[row+1])
				throw new IllegalArgumentException(name+" CSR row starts must be nondecreasing");
			for (int index : v) if (index < 1 || index > columns)
				throw new IllegalArgumentException(name+" CSR column index outside 1..columns");
		}
		private RuntimeValue integrateOneDimension(Context context) {
			String function = functionReference(arguments.get(0), name);
			requireDataArgument(context, 4); requireDataArgument(context, 5);
			if (arguments.size() == 7) requireDataArgument(context, 6);
			Diff lower = arguments.get(1).eval(context), upper = arguments.get(2).eval(context);
			RuntimeValue parameters = arguments.get(3).evalValue(context);
			RuntimeValue realData = arguments.get(4).evalValue(context);
			RuntimeValue intData = arguments.get(5).evalValue(context);
			if (parameters.kind != ValueKind.SCALAR || parameters.arrayRank != 1
					|| realData.kind != ValueKind.SCALAR || realData.arrayRank != 1
					|| intData.kind != ValueKind.SCALAR || intData.arrayRank != 1)
				throw new IllegalArgumentException("integrate_1d expects real parameter/data arrays and an int data array");
			double relative = arguments.size() == 7 ? arguments.get(6).eval(context).value : 1e-8;
			if (!(relative > 0) || !Double.isFinite(relative))
				throw new IllegalArgumentException("integrate_1d relative tolerance must be finite and positive");
			double[] theta = values(parameters.values), xReal = values(realData.values), xInteger = values(intData.values);
			double integral, lowerDerivative, upperDerivative; double[] thetaDerivative = new double[theta.length];
			Diff.suspendReverse();
			try {
				integral = integrateFunction(function, context, lower.value, upper.value, theta, xReal, xInteger, relative);
				for (int parameter = 0; parameter < theta.length; parameter++) {
					double step = 1e-6 * Math.max(1.0, Math.abs(theta[parameter]));
					double[] plus = theta.clone(), minus = theta.clone(); plus[parameter] += step; minus[parameter] -= step;
					thetaDerivative[parameter] = (integrateFunction(function, context, lower.value, upper.value,
							plus, xReal, xInteger, relative) - integrateFunction(function, context, lower.value, upper.value,
							minus, xReal, xInteger, relative)) / (2 * step);
				}
				lowerDerivative = Double.isFinite(lower.value)
						? -integrand(function, context, lower.value, lower.value, upper.value, theta, xReal, xInteger) : 0;
				upperDerivative = Double.isFinite(upper.value)
						? integrand(function, context, upper.value, lower.value, upper.value, theta, xReal, xInteger) : 0;
			} finally { Diff.resumeReverse(); }
			Diff[] inputs = new Diff[theta.length + 2]; double[] partials = new double[inputs.length];
			inputs[0] = lower; inputs[1] = upper; partials[0] = lowerDerivative; partials[1] = upperDerivative;
			for (int i = 0; i < theta.length; i++) { inputs[i + 2] = parameters.values[i]; partials[i + 2] = thetaDerivative[i]; }
			return RuntimeValue.scalar(Diff.atomic(integral, inputs, partials));
		}
		private double integrateFunction(String function, Context context, double lower, double upper,
				double[] theta, double[] realData, double[] intData, double relative) {
			IntegrationResult result = Integrate.integrate(x -> integrand(function, context, x, lower, upper,
					theta, realData, intData), lower, upper, relative, relative, 1000);
			if (!result.isSuccess()) throw new ArithmeticException("integrate_1d failed with status " + result.ier);
			return result.result;
		}
		private double integrand(String function, Context context, double x, double lower, double upper,
				double[] theta, double[] realData, double[] intData) {
			double complement = Double.isFinite(lower) && Double.isFinite(upper)
					? Math.min(Math.abs(x - lower), Math.abs(upper - x)) : Double.NaN;
			RuntimeValue result = invokeHigherOrder(function, context,
					plain(new double[] {x}, ValueKind.SCALAR, 0),
					plain(new double[] {complement}, ValueKind.SCALAR, 0),
					plain(theta, ValueKind.SCALAR, 1), plain(realData, ValueKind.SCALAR, 1),
					plain(intData, ValueKind.SCALAR, 1));
			return result.scalar(function + " integrand return").value;
		}
		private RuntimeValue algebraSolver(Context context) {
			String function = functionReference(arguments.get(0), name);
			requireDataArgument(context, 1); requireDataArgument(context, 3); requireDataArgument(context, 4);
			RuntimeValue initial = arguments.get(1).evalValue(context); requireKind(initial, ValueKind.VECTOR, name);
			RuntimeValue parameters = arguments.get(2).evalValue(context); requireKind(parameters, ValueKind.VECTOR, name);
			RuntimeValue realData = arguments.get(3).evalValue(context);
			RuntimeValue intData = arguments.get(4).evalValue(context);
			double[] initialValues = values(initial.values), parameterValues = values(parameters.values);
			double[] realValues = values(realData.values), intValues = values(intData.values);
			SensitivityResult solved;
			Diff.suspendReverse();
			try {
				solved = AlgebraicSolver.solveWithSensitivities((state, theta, ignored, residual) -> {
					RuntimeValue result = invokeHigherOrder(function, context,
							plain(state, ValueKind.VECTOR, 0), plain(theta, ValueKind.VECTOR, 0),
							plain(realValues, ValueKind.SCALAR, 1), plain(intValues, ValueKind.SCALAR, 1));
					requireKind(result, ValueKind.VECTOR, function); copyPlain(result, residual, function);
				}, initialValues, parameterValues, new double[0], AlgebraicSolver.Options.defaults());
			} finally { Diff.resumeReverse(); }
			double[] roots = solved.values()[0]; double[][] sensitivity = solved.sensitivities()[0];
			Diff[] result = new Diff[roots.length];
			for (int row = 0; row < result.length; row++) result[row] = Diff.atomic(roots[row], parameters.values, sensitivity[row]);
			return new RuntimeValue(result, new int[] {result.length}, ValueKind.VECTOR, 0);
		}
		private RuntimeValue odeSolver(Context context, boolean stiff) {
			String function = functionReference(arguments.get(0), name);
			requireDataArgument(context, 2); requireDataArgument(context, 3);
			requireDataArgument(context, 5); requireDataArgument(context, 6);
			RuntimeValue initial = arguments.get(1).evalValue(context); requireKind(initial, ValueKind.VECTOR, name);
			double initialTime = arguments.get(2).eval(context).value;
			double[] times = values(arguments.get(3).evalValue(context).values);
			RuntimeValue parameters = arguments.get(4).evalValue(context); requireKind(parameters, ValueKind.VECTOR, name);
			double[] realData = values(arguments.get(5).evalValue(context).values);
			double[] intData = values(arguments.get(6).evalValue(context).values);
			double[] initialValues = values(initial.values), parameterValues = values(parameters.values);
			jdistlib.inference.solver.OdeSystem system = (time, state, theta, ignored, derivative) -> {
				RuntimeValue result = invokeHigherOrder(function, context,
						plain(new double[] {time}, ValueKind.SCALAR, 0), plain(state, ValueKind.VECTOR, 0),
						plain(theta, ValueKind.VECTOR, 0), plain(realData, ValueKind.SCALAR, 1),
						plain(intData, ValueKind.SCALAR, 1));
				requireKind(result, ValueKind.VECTOR, function); copyPlain(result, derivative, function);
			};
			double[][] trajectory; double[][][] thetaSensitivity; double[][][] initialSensitivity;
			Diff.suspendReverse();
			try {
				if (stiff) {
					trajectory = StiffOdeSolver.integrate(system, initialValues, initialTime, times,
							parameterValues, new double[0], StiffOdeSolver.Options.defaults());
					thetaSensitivity = finiteOdeParameterSensitivity(system, initialValues, initialTime,
							times, parameterValues, trajectory, true);
				} else {
					SensitivityResult solved = OdeSolver.integrateWithSensitivities(system, initialValues,
							initialTime, times, parameterValues, new double[0], OdeSolver.Options.defaults());
					trajectory = solved.values(); thetaSensitivity = solved.sensitivities();
				}
				initialSensitivity = finiteOdeInitialSensitivity(system, initialValues,
						initialTime, times, parameterValues, trajectory, stiff);
			} finally { Diff.resumeReverse(); }
			return solverTrajectory(trajectory, initialSensitivity, thetaSensitivity, initial, parameters);
		}
		private RuntimeValue daeSolver(Context context) {
			String function = functionReference(arguments.get(0), name);
			requireDataArgument(context, 3); requireDataArgument(context, 4);
			requireDataArgument(context, 6); requireDataArgument(context, 7);
			RuntimeValue initial = arguments.get(1).evalValue(context); requireKind(initial, ValueKind.VECTOR, name);
			RuntimeValue initialDerivative = arguments.get(2).evalValue(context); requireKind(initialDerivative, ValueKind.VECTOR, name);
			double initialTime = arguments.get(3).eval(context).value;
			double[] times = values(arguments.get(4).evalValue(context).values);
			RuntimeValue parameters = arguments.get(5).evalValue(context); requireKind(parameters, ValueKind.VECTOR, name);
			double[] realData = values(arguments.get(6).evalValue(context).values);
			double[] intData = values(arguments.get(7).evalValue(context).values);
			double[] initialValues = values(initial.values), derivativeValues = values(initialDerivative.values);
			double[] parameterValues = values(parameters.values);
			jdistlib.inference.solver.DaeSystem system = (time, state, derivative, theta, ignored, residual) -> {
				RuntimeValue result = invokeHigherOrder(function, context,
						plain(new double[] {time}, ValueKind.SCALAR, 0), plain(state, ValueKind.VECTOR, 0),
						plain(derivative, ValueKind.VECTOR, 0), plain(theta, ValueKind.VECTOR, 0),
						plain(realData, ValueKind.SCALAR, 1), plain(intData, ValueKind.SCALAR, 1));
				requireKind(result, ValueKind.VECTOR, function); copyPlain(result, residual, function);
			};
			double[][] trajectory; double[][][] parameterSensitivity; double[][][] initialSensitivity;
			Diff.suspendReverse();
			try {
				double[] consistency = new double[initialValues.length];
				system.residual(initialTime, initialValues, derivativeValues, parameterValues, new double[0], consistency);
				if (AlgebraicSolver.infinityNorm(consistency) > 1e-5)
					throw new IllegalArgumentException("dae initial derivative is inconsistent with the residual");
				SensitivityResult solved = DaeSolver.integrateWithSensitivities(system, initialValues,
						initialTime, times, parameterValues, new double[0], AlgebraicSolver.Options.defaults());
				trajectory = solved.values(); parameterSensitivity = solved.sensitivities();
				initialSensitivity = finiteDaeInitialSensitivity(system, initialValues,
						initialTime, times, parameterValues, trajectory);
			} finally { Diff.resumeReverse(); }
			return solverTrajectory(trajectory, initialSensitivity, parameterSensitivity, initial, parameters);
		}
		private RuntimeValue invokeHigherOrder(String function, Context context, RuntimeValue... values) {
			List<UserFunction> overloads = context.functions == null ? null : context.functions.get(function);
			if (overloads == null) throw new IllegalArgumentException("unknown higher-order function: "+function);
			for (UserFunction candidate : overloads) {
				if (candidate.arguments.length != values.length) continue; boolean compatible = true;
				for (int i = 0; i < values.length; i++) if (!candidate.arguments[i].type.accepts(values[i])) {
					compatible = false; break;
				}
				if (compatible) {
					boolean[] data = new boolean[values.length]; Arrays.fill(data, true);
					return candidate.invoke(values, data, context);
				}
			}
			throw new IllegalArgumentException("no matching higher-order signature for "+function);
		}
		private String functionReference(Expr expression, String solver) {
			if (!(expression instanceof VariableExpr) || !((VariableExpr)expression).indices.isEmpty())
				throw new IllegalArgumentException(solver+" first argument must be a function name");
			return ((VariableExpr)expression).name;
		}
		private void requireDataArgument(Context context, int index) {
			if (!arguments.get(index).dataOnly(context))
				throw new IllegalArgumentException(name + " argument " + (index + 1) + " must be data-only");
		}
		private RuntimeValue plain(double[] values, ValueKind kind, int arrayRank) {
			Diff[] result = new Diff[values.length];
			for (int i = 0; i < result.length; i++) result[i] = Diff.constant(values[i], 0);
			int[] shape = kind == ValueKind.SCALAR && arrayRank == 0 ? new int[0] : new int[] {values.length};
			return new RuntimeValue(result, shape, kind, arrayRank);
		}
		private void copyPlain(RuntimeValue value, double[] target, String function) {
			if (value.values.length != target.length) throw new IllegalArgumentException(function+" return length mismatch");
			for (int i = 0; i < target.length; i++) target[i] = value.values[i].value;
		}
		private void requireKind(RuntimeValue value, ValueKind kind, String function) {
			if (value.kind != kind || value.arrayRank != 0)
				throw new IllegalArgumentException(function+" requires "+kind.name().toLowerCase());
		}
		private double[][][] finiteOdeInitialSensitivity(jdistlib.inference.solver.OdeSystem system,
				double[] initial, double initialTime, double[] times, double[] parameters,
				double[][] baseline, boolean stiff) {
			double[][][] result = new double[times.length][initial.length][initial.length];
			for (int column = 0; column < initial.length; column++) {
				double[] shifted = initial.clone(); double step = 1e-6*Math.max(1.0,Math.abs(initial[column])); shifted[column] += step;
				double[][] value = stiff ? StiffOdeSolver.integrate(system, shifted, initialTime, times, parameters,
						new double[0], StiffOdeSolver.Options.defaults()) : OdeSolver.integrate(system, shifted,
						initialTime, times, parameters, new double[0], OdeSolver.Options.defaults());
				for (int output = 0; output < times.length; output++) for (int row = 0; row < initial.length; row++)
					result[output][row][column] = (value[output][row]-baseline[output][row])/step;
			}
			return result;
		}
		private double[][][] finiteOdeParameterSensitivity(jdistlib.inference.solver.OdeSystem system,
				double[] initial, double initialTime, double[] times, double[] parameters,
				double[][] baseline, boolean stiff) {
			double[][][] result = new double[times.length][initial.length][parameters.length];
			for (int column = 0; column < parameters.length; column++) {
				double[] shifted = parameters.clone(); double step = 1e-6*Math.max(1.0,Math.abs(parameters[column])); shifted[column] += step;
				double[][] value = stiff ? StiffOdeSolver.integrate(system, initial, initialTime, times, shifted,
						new double[0], StiffOdeSolver.Options.defaults()) : OdeSolver.integrate(system, initial,
						initialTime, times, shifted, new double[0], OdeSolver.Options.defaults());
				for (int output = 0; output < times.length; output++) for (int row = 0; row < initial.length; row++)
					result[output][row][column] = (value[output][row]-baseline[output][row])/step;
			}
			return result;
		}
		private double[][][] finiteDaeInitialSensitivity(jdistlib.inference.solver.DaeSystem system,
				double[] initial, double initialTime, double[] times, double[] parameters, double[][] baseline) {
			double[][][] result = new double[times.length][initial.length][initial.length];
			for (int column = 0; column < initial.length; column++) {
				double[] shifted = initial.clone(); double step = 1e-6*Math.max(1.0,Math.abs(initial[column])); shifted[column] += step;
				double[][] value = DaeSolver.integrate(system, shifted, initialTime, times, parameters,
						new double[0], AlgebraicSolver.Options.defaults());
				for (int output = 0; output < times.length; output++) for (int row = 0; row < initial.length; row++)
					result[output][row][column] = (value[output][row]-baseline[output][row])/step;
			}
			return result;
		}
		private RuntimeValue solverTrajectory(double[][] values, double[][][] initialSensitivity,
				double[][][] parameterSensitivity, RuntimeValue initial, RuntimeValue parameters) {
			int outputs = values.length, states = initial.values.length;
			Diff[] result = new Diff[outputs*states]; Diff[] inputs = new Diff[states+parameters.values.length];
			System.arraycopy(initial.values,0,inputs,0,states);
			System.arraycopy(parameters.values,0,inputs,states,parameters.values.length);
			for (int output = 0; output < outputs; output++) for (int state = 0; state < states; state++) {
				double[] partials = new double[inputs.length];
				System.arraycopy(initialSensitivity[output][state],0,partials,0,states);
				System.arraycopy(parameterSensitivity[output][state],0,partials,states,parameters.values.length);
				result[output*states+state] = Diff.atomic(values[output][state],inputs,partials);
			}
			return new RuntimeValue(result,new int[] {outputs,states},ValueKind.VECTOR,1);
		}
		private int checkedPositiveSize(double value, String function) {
			if (value != Math.rint(value) || value < 1 || value > 100000)
				throw new IllegalArgumentException(function + " dimension must be a practical positive integer");
			return (int) value;
		}
		private Diff[] constants(int size, double value, Context context) {
			Diff[] result = new Diff[size];
			for (int i = 0; i < size; i++) result[i] = Diff.constant(value, context.dimension);
			return result;
		}
		private void requireVector(RuntimeValue value, String function) {
			if (value.arrayRank != 0 || value.kind != ValueKind.VECTOR && value.kind != ValueKind.ROW_VECTOR)
				throw new IllegalArgumentException(function + " requires a vector or row_vector, found "
						+ value.kind + Arrays.toString(value.shape));
		}
		private void requireMatrix(RuntimeValue value, String function) {
			if (value.arrayRank != 0 || value.kind != ValueKind.MATRIX)
				throw new IllegalArgumentException(function + " requires a matrix, found "
						+ value.kind + Arrays.toString(value.shape));
		}
		private void requireSquareMatrix(RuntimeValue value, String function) {
			requireMatrix(value, function);
			if (value.rows() != value.columns()) throw new IllegalArgumentException(function
					+ " requires a square matrix, found " + value.rows() + "x" + value.columns());
		}
		private RuntimeValue cholesky(RuntimeValue matrix, Context context) {
			int size = matrix.rows(); Diff[] lower = constants(size * size, 0.0, context);
			for (int row = 0; row < size; row++) for (int column = 0; column <= row; column++) {
				if (Math.abs(matrix.values[row * size + column].value
						- matrix.values[column * size + row].value) > 1e-8)
					throw new IllegalArgumentException("cholesky_decompose requires a symmetric matrix");
				Diff value = matrix.values[row * size + column];
				for (int k = 0; k < column; k++)
					value = value.subtract(lower[row * size + k].multiply(lower[column * size + k]));
				if (row == column) {
					if (!(value.value > 0.0)) throw new IllegalArgumentException("matrix is not positive definite");
					lower[row * size + column] = value.sqrt();
				} else lower[row * size + column] = value.divide(lower[column * size + column]);
			}
			return new RuntimeValue(lower, matrix.shape, ValueKind.MATRIX, 0);
		}
		private static final class QrPair {
			final RuntimeValue q, r;
			QrPair(RuntimeValue q, RuntimeValue r) { this.q = q; this.r = r; }
		}
		private QrPair qrThin(RuntimeValue matrix, Context context) {
			int rows = matrix.rows(), columns = matrix.columns();
			if (rows < columns) throw new IllegalArgumentException(name + " requires rows >= columns");
			Diff[] q = constants(rows * columns, 0.0, context);
			Diff[] r = constants(columns * columns, 0.0, context);
			for (int column = 0; column < columns; column++) {
				Diff[] work = new Diff[rows];
				for (int row = 0; row < rows; row++) work[row] = matrix.values[row * columns + column];
				for (int previous = 0; previous < column; previous++) {
					Diff projection = Diff.constant(0.0, context.dimension);
					for (int row = 0; row < rows; row++)
						projection = projection.add(q[row * columns + previous].multiply(work[row]));
					r[previous * columns + column] = projection;
					for (int row = 0; row < rows; row++)
						work[row] = work[row].subtract(q[row * columns + previous].multiply(projection));
				}
				Diff squaredNorm = Diff.constant(0.0, context.dimension);
				for (int row = 0; row < rows; row++) squaredNorm = squaredNorm.add(work[row].multiply(work[row]));
				if (!(squaredNorm.value > 0.0)) throw new IllegalArgumentException(name + " requires full column rank");
				Diff norm = squaredNorm.sqrt(); r[column * columns + column] = norm;
				for (int row = 0; row < rows; row++) q[row * columns + column] = work[row].divide(norm);
			}
			return new QrPair(new RuntimeValue(q, new int[] {rows, columns}, ValueKind.MATRIX, 0),
					new RuntimeValue(r, new int[] {columns, columns}, ValueKind.MATRIX, 0));
		}
		private RuntimeValue inverse(RuntimeValue matrix, Context context) {
			int size = matrix.rows(); Diff[][] augmented = new Diff[size][size * 2];
			for (int row = 0; row < size; row++) for (int column = 0; column < size * 2; column++)
				augmented[row][column] = column < size ? matrix.values[row * size + column]
						: Diff.constant(column - size == row ? 1.0 : 0.0, context.dimension);
			for (int pivot = 0; pivot < size; pivot++) {
				int best = pivot;
				for (int row = pivot + 1; row < size; row++)
					if (Math.abs(augmented[row][pivot].value) > Math.abs(augmented[best][pivot].value)) best = row;
				if (augmented[best][pivot].value == 0.0) throw new IllegalArgumentException("matrix is singular");
				Diff[] swap = augmented[pivot]; augmented[pivot] = augmented[best]; augmented[best] = swap;
				Diff divisor = augmented[pivot][pivot];
				for (int column = 0; column < size * 2; column++) augmented[pivot][column] = augmented[pivot][column].divide(divisor);
				for (int row = 0; row < size; row++) if (row != pivot) {
					Diff factor = augmented[row][pivot];
					for (int column = 0; column < size * 2; column++)
						augmented[row][column] = augmented[row][column].subtract(factor.multiply(augmented[pivot][column]));
				}
			}
			Diff[] result = new Diff[size * size];
			for (int row = 0; row < size; row++) for (int column = 0; column < size; column++)
				result[row * size + column] = augmented[row][size + column];
			return new RuntimeValue(result, matrix.shape, ValueKind.MATRIX, 0);
		}
		private Diff determinant(RuntimeValue matrix, Context context) {
			int size = matrix.rows(); Diff[][] values = new Diff[size][size];
			for (int row = 0; row < size; row++)
				System.arraycopy(matrix.values, row * size, values[row], 0, size);
			Diff result = Diff.constant(1.0, context.dimension); int sign = 1;
			for (int pivot = 0; pivot < size; pivot++) {
				int best = pivot;
				for (int row = pivot + 1; row < size; row++)
					if (Math.abs(values[row][pivot].value) > Math.abs(values[best][pivot].value)) best = row;
				if (values[best][pivot].value == 0.0) return Diff.constant(0.0, context.dimension);
				if (best != pivot) { Diff[] swap = values[pivot]; values[pivot] = values[best]; values[best] = swap; sign = -sign; }
				Diff diagonal = values[pivot][pivot]; result = result.multiply(diagonal);
				for (int row = pivot + 1; row < size; row++) {
					Diff factor = values[row][pivot].divide(diagonal);
					for (int column = pivot + 1; column < size; column++)
						values[row][column] = values[row][column].subtract(factor.multiply(values[pivot][column]));
				}
			}
			return sign < 0 ? result.negate() : result;
		}
		private RuntimeValue solveSpd(RuntimeValue matrix, RuntimeValue right, Context context) {
			RuntimeValue lower = cholesky(matrix, context); int size = matrix.rows();
			int columns; ValueKind kind; int[] shape;
			if (right.kind == ValueKind.VECTOR && right.arrayRank == 0) {
				if (right.values.length != size) throw new IllegalArgumentException("mdivide_left_spd RHS dimension mismatch");
				columns = 1; kind = ValueKind.VECTOR; shape = new int[] {size};
			} else if (right.kind == ValueKind.MATRIX && right.arrayRank == 0) {
				if (right.rows() != size) throw new IllegalArgumentException("mdivide_left_spd RHS row mismatch");
				columns = right.columns(); kind = ValueKind.MATRIX; shape = right.shape;
			} else throw new IllegalArgumentException("mdivide_left_spd RHS must be vector or matrix");
			Diff[] intermediate = new Diff[right.values.length], result = new Diff[right.values.length];
			for (int column = 0; column < columns; column++) for (int row = 0; row < size; row++) {
				Diff value = right.values[row * columns + column];
				for (int k = 0; k < row; k++) value = value.subtract(lower.values[row * size + k]
						.multiply(intermediate[k * columns + column]));
				intermediate[row * columns + column] = value.divide(lower.values[row * size + row]);
			}
			for (int column = 0; column < columns; column++) for (int row = size - 1; row >= 0; row--) {
				Diff value = intermediate[row * columns + column];
				for (int k = row + 1; k < size; k++) value = value.subtract(lower.values[k * size + row]
						.multiply(result[k * columns + column]));
				result[row * columns + column] = value.divide(lower.values[row * size + row]);
			}
			return new RuntimeValue(result, shape, kind, 0);
		}
		private Diff multiNormal(Context context, boolean suppliedCholesky) {
			RuntimeValue observations = arguments.get(0).evalValue(context);
			RuntimeValue location = arguments.get(1).evalValue(context);
			RuntimeValue matrix = arguments.get(2).evalValue(context);
			requireVector(observations, name); requireVector(location, name); requireSquareMatrix(matrix, name);
			int size = observations.values.length;
			if (location.values.length != size || matrix.rows() != size)
				throw new IllegalArgumentException(name + " arguments have incompatible dimensions");
			RuntimeValue lower = suppliedCholesky ? matrix : cholesky(matrix, context);
			Diff quadratic = Diff.constant(0.0, context.dimension);
			Diff logScale = Diff.constant(0.0, context.dimension);
			Diff[] solved = new Diff[size];
			for (int row = 0; row < size; row++) {
				Diff value = observations.values[row].subtract(location.values[row]);
				for (int k = 0; k < row; k++) value = value.subtract(lower.values[row * size + k].multiply(solved[k]));
				solved[row] = value.divide(lower.values[row * size + row]);
				quadratic = quadratic.add(solved[row].multiply(solved[row]));
				logScale = logScale.add(lower.values[row * size + row].log());
			}
			return quadratic.multiply(-0.5).subtract(logScale).add(-0.5 * size * Math.log(2.0 * Math.PI));
		}
		@Override public Diff eval(Context context) {
			RuntimeValue structured = structuredFunction(context);
			if (structured != null) return structured.scalar(name + " result");
			RuntimeValue userDefined = callUserFunction(context);
			if (userDefined != null) return userDefined.scalar(name + " result");
			Diff aggregate = aggregateFunction(context);
			if (aggregate != null) return aggregate;
			if ((name.endsWith("_lpdf") || name.endsWith("_lpmf")
					|| name.endsWith("_lupdf") || name.endsWith("_lupmf"))
					&& (context.functions == null || !context.functions.containsKey(name)))
				return vectorizedLogProbability(context);
			Diff[] values = new Diff[arguments.size()];
			for (int i = 0; i < values.length; i++) values[i] = arguments.get(i).eval(context);
			Diff scalar = scalarFunction(name, values, context);
			if (scalar != null) return scalar;
			if (name.endsWith("_rng")) return random(name.substring(0, name.length() - 4), values, context);
			throw new IllegalArgumentException("unknown function: " + name);
		}
		@Override public Diff[] evalVector(Context context) {
			RuntimeValue structured = structuredFunction(context);
			if (structured != null) return structured.values;
			RuntimeValue userDefined = callUserFunction(context);
			if (userDefined != null) return userDefined.values;
			RuntimeValue broadcast = broadcastScalarFunction(context);
			if (broadcast != null) return broadcast.values;
			if ((name.equals("to_vector") || name.equals("to_row_vector")
					|| name.equals("to_array_1d")) && arguments.size() == 1)
				return arguments.get(0).evalVector(context);
			if ((name.equals("rep_vector") || name.equals("rep_row_vector")
					|| name.equals("rep_array")) && arguments.size() == 2) {
				Diff value = arguments.get(0).eval(context);
				int length = checkedLoopBound(arguments.get(1).eval(context).value, name);
				if (length < 0 || length > 1000000)
					throw new IllegalArgumentException(name + " length must be nonnegative and practical");
				Diff[] result = new Diff[length]; Arrays.fill(result, value); return result;
			}
			if (arguments.size() == 1 && !name.endsWith("_rng")) {
				Diff[] input = arguments.get(0).evalVector(context);
				if (input.length > 1) {
					Diff[] result = new Diff[input.length];
					for (int i = 0; i < input.length; i++) {
						result[i] = scalarFunction(name, new Diff[] {input[i]}, context);
						if (result[i] == null) return new Diff[] {eval(context)};
					}
					return result;
				}
			}
			return new Diff[] {eval(context)};
		}
		private Diff aggregateFunction(Context context) {
			if ((name.equals("sum") || name.equals("prod") || name.equals("mean"))
					&& arguments.size() == 1) {
				Diff[] values = arguments.get(0).evalVector(context);
				Diff result = Diff.constant(name.equals("prod") ? 1.0 : 0.0, context.dimension);
				for (Diff value : values) result = name.equals("prod")
						? result.multiply(value) : result.add(value);
				return name.equals("mean") ? result.multiply(1.0 / values.length) : result;
			}
			if (name.equals("dot_product") && arguments.size() == 2) {
				Diff[] leftValues = arguments.get(0).evalVector(context);
				Diff[] rightValues = arguments.get(1).evalVector(context);
				if (leftValues.length != rightValues.length)
					throw new IllegalArgumentException("dot_product arguments must have equal lengths");
				Diff[] inputs = new Diff[leftValues.length * 2];
				double[] partials = new double[inputs.length]; double value = 0.0;
				for (int i = 0; i < leftValues.length; i++) {
					inputs[i] = leftValues[i]; inputs[leftValues.length + i] = rightValues[i];
					partials[i] = rightValues[i].value; partials[leftValues.length + i] = leftValues[i].value;
					value += leftValues[i].value * rightValues[i].value;
				}
				return Diff.atomic(value, inputs, partials);
			}
			if ((name.equals("num_elements") || name.equals("size") || name.equals("rows")
					|| name.equals("cols")) && arguments.size() == 1) {
				RuntimeValue value = arguments.get(0).evalValue(context);
				if (name.equals("num_elements"))
					return Diff.constant(value.logicalSize(), context.dimension);
				int[] shape = value.shape;
				int extent = name.equals("cols") ? shape[shape.length - 1]
						: name.equals("rows") && value.kind == ValueKind.MATRIX ? shape[shape.length - 2]
						: shape.length == 0 ? 1 : shape[0];
				return Diff.constant(extent, context.dimension);
			}
			return null;
		}
		private RuntimeValue callUserFunction(Context context) {
			if (context.functions == null) return null;
			List<UserFunction> overloads = context.functions.get(name);
			if (overloads == null) return null;
			RuntimeValue[] values = new RuntimeValue[arguments.size()];
			boolean[] dataArguments = new boolean[arguments.size()];
			for (int i = 0; i < values.length; i++) {
				values[i] = arguments.get(i).evalValue(context);
				dataArguments[i] = arguments.get(i).dataOnly(context);
			}
			UserFunction selected = null; int selectedPromotions = Integer.MAX_VALUE;
			for (UserFunction candidate : overloads) {
				if (candidate.arguments.length != values.length) continue;
				int promotions = 0; boolean compatible = true;
				for (int i = 0; i < values.length; i++) {
					FunctionArgument formal = candidate.arguments[i]; RuntimeValue actual = values[i];
					if (!formal.type.accepts(actual)) {
						compatible = false; break;
					}
					boolean integerArgument = isIntegerArgument(arguments.get(i), actual, context);
					if (formal.type.integer && !integerArgument) { compatible = false; break; }
					if (!formal.type.integer && integerArgument) promotions++;
					if (formal.dataOnly && !dataArguments[i]) { compatible = false; break; }
				}
				if (compatible && promotions == selectedPromotions)
					throw new IllegalArgumentException("ambiguous overload for user function " + name);
				if (compatible && promotions < selectedPromotions) {
					selected = candidate; selectedPromotions = promotions;
				}
			}
			if (selected == null)
				throw new IllegalArgumentException("no matching overload for user function " + name);
			return selected.invoke(values, dataArguments, context);
		}
		private boolean isIntegerArgument(Expr expression, RuntimeValue value, Context context) {
			for (Diff element : value.values) if (element.value != Math.rint(element.value)) return false;
			return isIntegerExpression(expression, context);
		}
		private boolean isIntegerExpression(Expr expression, Context context) {
			if (expression instanceof NumberExpr)
				return ((NumberExpr) expression).value == Math.rint(((NumberExpr) expression).value);
			if (expression instanceof VariableExpr) {
				String variable = ((VariableExpr) expression).name;
				if (context.isIntegerLocal(variable)) return true;
				if (context.state != null && context.state.hasParameter(variable)) return false;
				double[] data = context.data.get(variable);
				if (data == null) return false;
				for (double element : data) if (element != Math.rint(element)) return false;
				return true;
			}
			if (expression instanceof UnaryExpr)
				return isIntegerExpression(((UnaryExpr) expression).operand, context);
			if (expression instanceof BinaryExpr) {
				BinaryExpr binary = (BinaryExpr) expression;
				if (binary.operator.equals("/") || binary.operator.equals("./")
						|| binary.operator.equals("^") || binary.operator.equals(".*")) return false;
				if (binary.operator.equals("<") || binary.operator.equals("<=") || binary.operator.equals(">")
						|| binary.operator.equals(">=") || binary.operator.equals("==")
						|| binary.operator.equals("!=") || binary.operator.equals("&&") || binary.operator.equals("||"))
					return true;
				return isIntegerExpression(binary.left, context) && isIntegerExpression(binary.right, context);
			}
			if (expression instanceof ConditionalExpr) {
				ConditionalExpr conditional = (ConditionalExpr) expression;
				return isIntegerExpression(conditional.whenTrue, context)
						&& isIntegerExpression(conditional.whenFalse, context);
			}
			return false;
		}
		private boolean constantGradient(RuntimeValue value) {
			for (Diff element : value.values) if (!element.constantGradient()) return false;
			return true;
		}
		private Diff vectorizedLogProbability(Context context) {
			int suffixLength = name.endsWith("_lupdf") || name.endsWith("_lupmf") ? 6 : 5;
			String distribution = name.substring(0, name.length() - suffixLength);
			RuntimeValue[] values = new RuntimeValue[arguments.size()];
			RuntimeValue template = null; int length = 1;
			for (int i = 0; i < arguments.size(); i++) {
				values[i] = arguments.get(i).evalValue(context);
				if (!values[i].isScalar()) {
					if (template != null && template.values.length != values[i].values.length)
						throw new IllegalArgumentException("incompatible probability broadcast: "
								+ template.kind + Arrays.toString(template.shape) + " versus "
								+ values[i].kind + Arrays.toString(values[i].shape));
					template = values[i]; length = values[i].values.length;
				}
			}
			Diff result = Diff.constant(0.0, context.dimension);
			for (int element = 0; element < length; element++) {
				Diff[] scalar = new Diff[values.length];
				for (int i = 0; i < values.length; i++)
					scalar[i] = values[i].values[values[i].isScalar() ? 0 : element];
				result = result.add(logProbability(distribution, scalar));
			}
			return result;
		}
		@Override public void collect(Set<String> names) { for (Expr argument : arguments) argument.collect(names); }
		@Override public boolean dataOnly(Context context) {
			if (name.endsWith("_rng") || name.endsWith("_lp")) return false;
			for (Expr argument : arguments) if (!argument.dataOnly(context)) return false;
			return true;
		}
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
		final ModelState state; final Map<String, double[]> data;
		final Map<String, int[]> shapes; final Map<String, ValueType> types; final int dimension;
		final RandomEngine random;
		final Map<String, List<UserFunction>> functions;
		int functionDepth;
		Diff functionTarget;
		final List<String> functionNames = new ArrayList<String>();
		final List<Map<String, RuntimeValue>> scopes = new ArrayList<Map<String, RuntimeValue>>();
		final List<Set<String>> integerScopes = new ArrayList<Set<String>>();
		final List<Set<String>> dataScopes = new ArrayList<Set<String>>();
		Context(ModelState state, Map<String, double[]> data, Map<String, int[]> shapes,
				Map<String, ValueType> types,
				int dimension, RandomEngine random, Map<String, List<UserFunction>> functions) {
			this.state = state; this.data = data; this.shapes = shapes; this.types = types; this.dimension = dimension;
			this.random = random; this.functions = functions;
			functionTarget = Diff.constant(0.0, dimension);
			scopes.add(new LinkedHashMap<String, RuntimeValue>());
			integerScopes.add(new LinkedHashSet<String>());
			dataScopes.add(new LinkedHashSet<String>());
		}
		void pushScope() {
			scopes.add(new LinkedHashMap<String, RuntimeValue>());
			integerScopes.add(new LinkedHashSet<String>());
			dataScopes.add(new LinkedHashSet<String>());
		}
		void popScope() {
			scopes.remove(scopes.size() - 1); integerScopes.remove(integerScopes.size() - 1);
			dataScopes.remove(dataScopes.size() - 1);
		}
		RuntimeValue local(String name) {
			for (int i = scopes.size() - 1; i >= 0; i--) {
				RuntimeValue value = scopes.get(i).get(name); if (value != null) return value;
			}
			return null;
		}
		void declareLocal(String name, Diff[] value, boolean integer) {
			declareLocal(name, value.length == 1 ? RuntimeValue.scalar(value[0])
					: new RuntimeValue(value, new int[] {value.length}, ValueKind.VECTOR, 0), integer, false);
		}
		void declareLocal(String name, RuntimeValue value, boolean integer) {
			declareLocal(name, value, integer, false);
		}
		void declareLocal(String name, RuntimeValue value, boolean integer, boolean dataOnly) {
			Map<String, RuntimeValue> scope = scopes.get(scopes.size() - 1);
			if (scope.containsKey(name)) throw new IllegalArgumentException("duplicate local variable: " + name);
			scope.put(name, value);
			if (integer) integerScopes.get(integerScopes.size() - 1).add(name);
			if (dataOnly) dataScopes.get(dataScopes.size() - 1).add(name);
		}
		void setLocal(String name, Diff[] value) { scopes.get(0).put(name, wrap(name, value)); }
		void setLocal(String name, RuntimeValue value) { scopes.get(0).put(name, value); }
		void setLocal(String name, RuntimeValue value, boolean dataOnly) {
			scopes.get(0).put(name, value); if (dataOnly) dataScopes.get(0).add(name);
		}
		RuntimeValue wrap(String name, Diff[] values) {
			int[] shape = shape(name, values.length);
			ValueType declared = types == null ? null : types.get(name);
			ValueKind kind = declared == null ? shape.length == 0 ? ValueKind.SCALAR
					: shape.length == 1 ? ValueKind.VECTOR : ValueKind.MATRIX : declared.kind;
			return new RuntimeValue(values, shape, kind, declared == null ? 0 : declared.arrayRank);
		}
		int[] shape(String name, int length) {
			int[] declared = shapes == null ? null : shapes.get(name);
			return declared == null ? new int[] {length} : declared;
		}
		void assignLocal(String name, Diff[] value) {
			assignLocal(name, value.length == 1 ? RuntimeValue.scalar(value[0])
					: new RuntimeValue(value, new int[] {value.length}, ValueKind.VECTOR, 0));
		}
		void assignLocal(String name, RuntimeValue value) {
			for (int i = scopes.size() - 1; i >= 0; i--) {
				Map<String, RuntimeValue> scope = scopes.get(i);
				if (scope.containsKey(name)) {
					if (integerScopes.get(i).contains(name)
							&& (value.values.length != 1 || value.values[0].value != Math.rint(value.values[0].value)))
						throw new IllegalArgumentException("integer local '" + name
								+ "' received a non-integer value");
					scope.put(name, value); return;
				}
			}
			throw new IllegalArgumentException("assignment requires a declared local variable: " + name);
		}
		Diff requireScalarLocal(String name) {
			RuntimeValue value = local(name);
			if (value == null || value.values.length != 1)
				throw new IllegalArgumentException("scalar local variable expected: " + name);
			return value.values[0];
		}
		boolean isIntegerLocal(String name) {
			for (int i = integerScopes.size() - 1; i >= 0; i--)
				if (integerScopes.get(i).contains(name)) return true;
			return false;
		}
		boolean isDataVariable(String name) {
			for (int i = dataScopes.size() - 1; i >= 0; i--)
				if (dataScopes.get(i).contains(name)) return true;
			if (state != null && state.hasParameter(name)) return false;
			return data.containsKey(name);
		}
		void addFunctionTarget(Diff contribution) {
			if (functionNames.isEmpty() || !functionNames.get(functionNames.size() - 1).endsWith("_lp"))
				throw new IllegalArgumentException("target changes inside a function require the _lp suffix");
			functionTarget = functionTarget.add(contribution);
		}
		Diff consumeFunctionTarget() {
			Diff result = functionTarget; functionTarget = Diff.constant(0.0, dimension); return result;
		}
	}
	private static final class Diff {
		private static final class ReverseContext {
			ReverseTape tape; int dimension; int[] handles = new int[128];
			int[] coordinates = new int[128]; int handleCount; int suspensionDepth;
			void begin(ReverseTape value, int dimensions) {
				if (tape != null) throw new IllegalStateException("nested reverse evaluation");
				tape = value; dimension = dimensions; handleCount = 0;
			}
			void register(int handle, int coordinate) {
				if (handleCount == handles.length) {
					handles = Arrays.copyOf(handles, handles.length * 2);
					coordinates = Arrays.copyOf(coordinates, coordinates.length * 2);
				}
				handles[handleCount] = handle; coordinates[handleCount++] = coordinate;
			}
			void clear() { tape = null; dimension = 0; handleCount = 0; suspensionDepth = 0; }
			boolean active() { return tape != null && suspensionDepth == 0; }
		}
		private static final ThreadLocal<ReverseContext> REVERSE =
				new ThreadLocal<ReverseContext>() {
					@Override protected ReverseContext initialValue() { return new ReverseContext(); }
				};
		final double value; final double[] gradient; final int node;
		Diff(double value, double[] gradient) { this(value, gradient, -1); }
		Diff(double value, double[] gradient, int node) {
			this.value = value; this.gradient = gradient; this.node = node;
		}
		static void beginReverse(ReverseTape tape, int dimension) {
			REVERSE.get().begin(tape, dimension);
		}
		static void endReverse() { REVERSE.get().clear(); }
		static void suspendReverse() { REVERSE.get().suspensionDepth++; }
		static void resumeReverse() {
			ReverseContext reverse = REVERSE.get();
			if (reverse.suspensionDepth < 1) throw new IllegalStateException("reverse evaluation is not suspended");
			reverse.suspensionDepth--;
		}
		static Diff constant(double value, int dimension) {
			ReverseContext reverse = REVERSE.get();
			return !reverse.active() ? new Diff(value, new double[dimension])
					: new Diff(value, null, reverse.tape.constant(value));
		}
		static Diff variable(double value, int dimension, int coordinate) {
			ReverseContext reverse = REVERSE.get();
			if (!reverse.active()) {
				double[] derivative = new double[dimension]; derivative[coordinate] = 1.0;
				return new Diff(value, derivative);
			}
			int handle = reverse.tape.variable(value); reverse.register(handle, coordinate);
			return new Diff(value, null, handle);
		}
		static void reverseInto(Diff output, double[] gradient) {
			ReverseContext reverse = REVERSE.get();
			if (reverse.tape == null || output.node < 0)
				throw new IllegalStateException("reverse output is not on the active tape");
			reverse.tape.reverse(output.node);
			for (int i = 0; i < reverse.handleCount; i++)
				gradient[reverse.coordinates[i]] += reverse.tape.adjoint(reverse.handles[i]);
		}
		static Diff atomic(double value, Diff[] inputs, double[] partials) {
			ReverseContext reverse = REVERSE.get();
			if (!reverse.active()) {
				double[] derivative = new double[reverse.dimension];
				if (inputs.length > 0) derivative = new double[inputs[0].gradient.length];
				for (int input = 0; input < inputs.length; input++) for (int i = 0; i < derivative.length; i++)
					derivative[i] += partials[input] * inputs[input].gradient[i];
				return new Diff(value, derivative);
			}
			int[] parents = new int[inputs.length];
			for (int i = 0; i < inputs.length; i++) parents[i] = inputs[i].node;
			return new Diff(value, null, reverse.tape.atomic(value, parents, partials));
		}
		int dimension() { return gradient == null ? REVERSE.get().dimension : gradient.length; }
		Diff add(Diff other) { return combine(value + other.value, 1.0, other, 1.0); }
		Diff add(double other) { return scale(value + other, 1.0); }
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
			if (other.value == Math.rint(other.value) && (other.constantGradient() || value <= 0.0)) {
				double result = Math.pow(value, other.value);
				double baseDerivative = other.value == 0.0 ? 0.0
						: other.value * Math.pow(value, other.value - 1.0);
				return combine(result, baseDerivative, other, 0.0);
			}
			return log().multiply(other).exp();
		}
		private boolean constantGradient() {
			if (gradient == null) return false;
			for (double derivative : gradient) if (derivative != 0.0) return false;
			return true;
		}
		private Diff scale(double result, double multiplier) {
			if (gradient == null) {
				ReverseTape tape = REVERSE.get().tape;
				return new Diff(result, null, tape.customUnary(node, result, multiplier));
			}
			double[] derivative = new double[gradient.length];
			for (int i = 0; i < derivative.length; i++) derivative[i] = multiplier * gradient[i];
			return new Diff(result, derivative);
		}
		private Diff combine(double result, double ownMultiplier, Diff other, double otherMultiplier) {
			if (gradient == null) {
				ReverseTape tape = REVERSE.get().tape;
				return new Diff(result, null, tape.customBinary(node, other.node, result,
						ownMultiplier, otherMultiplier));
			}
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
			double difference = x[0].value - x[1].value;
			double inverseScale = 1.0 / x[2].value, z = difference * inverseScale;
			double value = -Math.log(x[2].value) - 0.5 * Math.log(2.0 * Math.PI) - 0.5 * z * z;
			return Diff.atomic(value, x, new double[] {-z * inverseScale, z * inverseScale,
					(z * z - 1.0) * inverseScale});
		}
		if (distribution.equals("lognormal") && x.length == 3) {
			if (!positive(x[0]) || !positive(x[2])) return outside(x);
			Diff logged = x[0].log(); Diff z = logged.subtract(x[1]).divide(x[2]);
			return x[2].log().negate().subtract(logged).add(-0.5 * Math.log(2.0 * Math.PI))
					.subtract(z.multiply(z).multiply(0.5));
		}
		if (distribution.equals("student_t") && x.length == 4) {
			if (!positive(x[1]) || !positive(x[3])) return outside(x);
			double nu = x[1].value, sigma = x[3].value;
			double z = (x[0].value - x[2].value) / sigma, z2 = z * z;
			double value = lgammafn(0.5 * (nu + 1.0)) - lgammafn(0.5 * nu)
					- 0.5 * Math.log(nu * Math.PI) - Math.log(sigma)
					- 0.5 * (nu + 1.0) * Math.log1p(z2 / nu);
			double locationDerivative = (nu + 1.0) * z / (sigma * (nu + z2));
			double nuDerivative = 0.5 * psi(0.5 * (nu + 1.0)) - 0.5 * psi(0.5 * nu)
					- 0.5 / nu - 0.5 * Math.log1p(z2 / nu)
					+ 0.5 * (nu + 1.0) * z2 / (nu * (nu + z2));
			return Diff.atomic(value, x, new double[] {-locationDerivative, nuDerivative,
					locationDerivative, (-1.0 + (nu + 1.0) * z2 / (nu + z2)) / sigma});
		}
		if (distribution.equals("cauchy") && x.length == 3) {
			if (!positive(x[2])) return outside(x);
			Diff z = x[0].subtract(x[1]).divide(x[2]);
			return x[2].log().negate().add(-Math.log(Math.PI))
					.subtract(Diff.constant(1.0, x[0].dimension()).add(z.multiply(z)).log());
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
			Diff shape = x[1].multiply(0.5); Diff rate = Diff.constant(0.5, x[0].dimension());
			return shape.multiply(rate.log()).subtract(shape.lgamma())
					.add(shape.subtract(one(x)).multiply(x[0].log())).subtract(rate.multiply(x[0]));
		}
		if (distribution.equals("inv_chi_square") && x.length == 2) {
			Diff[] expanded = {x[0], x[1].multiply(0.5), Diff.constant(0.5, x[0].dimension())};
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
				return x[0].value == 0.0 ? Diff.constant(0.0, x[0].dimension()) : outside(x);
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
		int dimension = values.length == 0 ? 0 : values[0].dimension();
		return Diff.constant(Double.NEGATIVE_INFINITY, dimension);
	}
	private static Diff one(Diff[] values) {
		return Diff.constant(1.0, values[0].dimension());
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
		return Diff.constant(value, argument.dimension());
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
				if (block.equals("functions")) parseFunction(program);
				else if (block.equals("data")) program.data.add(parseDeclaration());
				else if (block.equals("parameters")) program.parameters.add(parseDeclaration());
				else if (block.equals("transformed data")) program.transformedData.add(parseAssignment(true));
				else if (block.equals("transformed parameters")) program.transformedParameters.add(parseAssignment(true));
				else if (block.equals("generated quantities")) program.generated.add(parseAssignment(true));
				else if (block.equals("model")) program.model.add(parseStatement());
				else fail(start, "unknown block '" + block + "'");
			}
		}
		private void parseFunction(Program program) {
			FunctionType returnType = parseFunctionType(false);
			Token name = require(TokenKind.IDENTIFIER, "function name expected"); require("(");
			List<FunctionArgument> arguments = new ArrayList<FunctionArgument>();
			if (!current.text.equals(")")) {
				do {
					boolean dataOnly = current.text.equals("data"); if (dataOnly) advance();
					FunctionType type = parseFunctionType(true);
					Token argument = require(TokenKind.IDENTIFIER, "function argument name expected");
					arguments.add(new FunctionArgument(argument.text, type, dataOnly));
				} while (accept(","));
			}
			require(")");
			Statement body = accept(";") ? null : parseStatement();
			UserFunction function = new UserFunction(name.text, returnType,
					arguments.toArray(new FunctionArgument[arguments.size()]), body, name);
			validateProbabilityFunction(function, name);
			List<UserFunction> overloads = program.functions.get(name.text);
			if (overloads == null) {
				overloads = new ArrayList<UserFunction>(); program.functions.put(name.text, overloads);
			}
			for (UserFunction existing : overloads) {
				if (existing.sameSignature(function)) {
					if (!existing.returnType.sameSignature(function.returnType))
						fail(name, "definition return type differs from forward declaration");
					if (existing.body == null && function.body != null) {
						existing.body = function.body; return;
					}
					fail(name, existing.body == null ? "duplicate forward declaration"
							: "duplicate user-function signature");
				}
			}
			overloads.add(function);
		}
		private FunctionType parseFunctionType(boolean argument) {
			int arrayRank = 0;
			while (current.text.equals("array")) {
				advance(); require("["); arrayRank++;
				while (accept(",")) arrayRank++;
				require("]");
			}
			if (current.text.equals("tuple")) {
				if (arrayRank != 0) fail(current, "arrays of tuples are not supported as function values");
				return new FunctionType(parseUnsizedTupleType());
			}
			Token token = require(TokenKind.IDENTIFIER, "function type expected");
			if (!(token.text.equals("real") || token.text.equals("int") || token.text.equals("complex")
					|| token.text.equals("vector") || token.text.equals("row_vector") || token.text.equals("matrix")
					|| token.text.equals("complex_vector") || token.text.equals("complex_row_vector")
					|| token.text.equals("complex_matrix")))
				fail(token, "unsupported function type");
			if (accept("[")) fail(token, argument
					? "function argument container dimensions must be unsized"
					: "function return container dimensions must be unsized");
			return new FunctionType(valueKind(token.text), arrayRank, token.text.equals("int"));
		}
		private TupleTypeSpec parseUnsizedTupleType() {
			require("tuple"); require("("); List<TupleTypeSpec> members = new ArrayList<TupleTypeSpec>();
			members.add(parseUnsizedTupleMemberType());
			if (!accept(",")) fail(current, "tuple types require a comma-separated member list");
			if (!current.text.equals(")")) {
				do { members.add(parseUnsizedTupleMemberType()); } while (accept(","));
			}
			require(")");
			return new TupleTypeSpec(ValueKind.TUPLE, 0, Collections.<Expr>emptyList(),
					members.toArray(new TupleTypeSpec[members.size()]));
		}
		private TupleTypeSpec parseUnsizedTupleMemberType() {
			if (current.text.equals("tuple")) return parseUnsizedTupleType();
			int arrayRank = 0;
			while (current.text.equals("array")) {
				advance(); require("["); arrayRank++;
				while (accept(",")) arrayRank++;
				require("]");
			}
			Token type = require(TokenKind.IDENTIFIER, "tuple member function type expected");
			if (!(type.text.equals("real") || type.text.equals("int") || type.text.equals("complex")
					|| type.text.equals("vector") || type.text.equals("row_vector") || type.text.equals("matrix")
					|| type.text.equals("complex_vector") || type.text.equals("complex_row_vector")
					|| type.text.equals("complex_matrix"))) fail(type, "unsupported tuple member function type");
			if (accept("[")) fail(type, "tuple function members must be unsized");
			return new TupleTypeSpec(valueKind(type.text), arrayRank, Collections.<Expr>emptyList(), null);
		}
		private void validateProbabilityFunction(UserFunction function, Token token) {
			if (!(function.name.endsWith("_lpdf") || function.name.endsWith("_lpmf"))) return;
			if (function.returnType.integer || function.returnType.kind != ValueKind.SCALAR
					|| function.returnType.arrayRank != 0)
				fail(token, "probability functions must return real");
			if (function.arguments.length == 0) fail(token, "probability functions require a variate argument");
			boolean mass = function.name.endsWith("_lpmf");
			if (mass != function.arguments[0].type.integer)
				fail(token, mass ? "_lpmf first argument must be int or an int array"
						: "_lpdf first argument must be real-valued");
		}
		private Declaration parseDeclaration() {
			List<Expr> arrayShape = new ArrayList<Expr>();
			if (current.text.equals("array")) {
				advance(); require("["); arrayShape.addAll(dimensionList()); require("]");
			}
			Token typeToken = require(TokenKind.IDENTIFIER, "declaration type expected");
			String type = typeToken.text;
			if (!(type.equals("real") || type.equals("int") || type.equals("complex") || type.equals("vector")
					|| type.equals("row_vector") || type.equals("matrix")
					|| type.equals("complex_vector") || type.equals("complex_row_vector")
					|| type.equals("complex_matrix")
					|| type.equals("simplex") || type.equals("ordered")
					|| type.equals("positive_ordered")
					|| type.equals("sum_to_zero_vector") || type.equals("unit_vector")
					|| type.equals("cov_matrix") || type.equals("corr_matrix")
					|| type.equals("cholesky_factor_cov")
					|| type.equals("cholesky_factor_corr"))) fail(typeToken, "unsupported declaration type");
			Expr lower = null, upper = null, offset = null, multiplier = null;
			if (accept("<")) {
				do {
					Token bound = require(TokenKind.IDENTIFIER, "constraint name expected"); require("=");
					Expr value = expression(4);
					if (bound.text.equals("lower")) lower = value;
					else if (bound.text.equals("upper")) upper = value;
					else if (bound.text.equals("offset")) offset = value;
					else if (bound.text.equals("multiplier")) multiplier = value;
					else fail(bound, "unsupported declaration constraint");
				} while (accept(","));
				require(">");
			}
			List<Expr> baseShape = new ArrayList<Expr>();
			if (accept("[")) { baseShape.addAll(dimensionList()); require("]"); }
			int baseDimensions = type.equals("matrix") || type.equals("complex_matrix")
					|| type.equals("cholesky_factor_cov") ? 2
					: type.equals("vector") || type.equals("row_vector")
					|| type.equals("complex_vector") || type.equals("complex_row_vector")
					|| type.equals("simplex") || type.equals("ordered")
					|| type.equals("positive_ordered")
					|| type.equals("sum_to_zero_vector") || type.equals("unit_vector")
					|| type.equals("cov_matrix") || type.equals("corr_matrix")
					|| type.equals("cholesky_factor_corr") ? 1 : 0;
			if (baseShape.size() != baseDimensions)
				fail(typeToken, type + " requires " + baseDimensions + " dimension(s)");
			Token name = require(TokenKind.IDENTIFIER, "declaration name expected");
			if (accept("[")) { arrayShape.addAll(dimensionList()); require("]"); }
			require(";");
			List<Expr> shape = new ArrayList<Expr>(arrayShape); shape.addAll(baseShape);
			if (type.equals("cov_matrix") || type.equals("corr_matrix")
					|| type.equals("cholesky_factor_corr")) shape.add(baseShape.get(0));
			Expr dimension = product(shape);
			return new Declaration(type, name.text, dimension, lower, upper, offset,
					multiplier, shape, baseShape, arrayShape.size(), type.equals("int"), name);
		}
		private List<Expr> dimensionList() {
			List<Expr> result = new ArrayList<Expr>(); result.add(expression(0));
			while (accept(",")) result.add(expression(0));
			return result;
		}
		private Expr product(List<Expr> dimensions) {
			if (dimensions.isEmpty()) return null;
			Expr result = dimensions.get(0);
			for (int i = 1; i < dimensions.size(); i++)
				result = new BinaryExpr("*", result, dimensions.get(i));
			return result;
		}
		private Assignment parseAssignment(boolean declarationAllowed) {
			List<Expr> arrayShape = new ArrayList<Expr>();
			if (declarationAllowed && current.text.equals("array")) {
				advance(); require("["); arrayShape.addAll(dimensionList()); require("]");
			}
			String type = "real"; List<Expr> baseShape = new ArrayList<Expr>();
			if (declarationAllowed && current.kind == TokenKind.IDENTIFIER
					&& isAssignmentType(current.text)) {
				type = current.text; advance();
				if (accept("<")) {
					do {
						require(TokenKind.IDENTIFIER, "constraint name expected"); require("=");
						expression(4);
					} while (accept(","));
					require(">");
				}
				if (accept("[")) { baseShape.addAll(dimensionList()); require("]"); }
			}
			Token name = require(TokenKind.IDENTIFIER, "assignment name expected");
			if (accept("[")) { arrayShape.addAll(dimensionList()); require("]"); }
			List<Expr> shape = new ArrayList<Expr>(arrayShape); shape.addAll(baseShape);
			if (type.equals("cov_matrix") || type.equals("corr_matrix")
					|| type.equals("cholesky_factor_corr")) shape.add(baseShape.get(0));
			require("="); Expr value = expression(0); require(";");
			return new Assignment(name.text, value, shape, valueKind(type), arrayShape.size(), name);
		}
		private boolean isAssignmentType(String type) {
			return type.equals("real") || type.equals("int") || type.equals("complex") || type.equals("vector")
					|| type.equals("row_vector") || type.equals("matrix")
					|| type.equals("complex_vector") || type.equals("complex_row_vector")
					|| type.equals("complex_matrix")
					|| type.equals("simplex") || type.equals("ordered")
					|| type.equals("positive_ordered") || type.equals("sum_to_zero_vector")
					|| type.equals("unit_vector") || type.equals("cov_matrix")
					|| type.equals("corr_matrix") || type.equals("cholesky_factor_cov")
					|| type.equals("cholesky_factor_corr");
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
			if (current.text.equals("return")) {
				advance(); Expr value = expression(0); require(";");
				return new ReturnStatement(value);
			}
			if (current.text.equals("tuple")) return parseTupleLocalDeclaration();
			if (current.text.equals("array") || isAssignmentType(current.text)) {
				return parseLocalDeclaration();
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
			if (left instanceof TupleAccessExpr) {
				if(!current.text.equals("="))fail(current,"tuple members currently support direct assignment");
				advance();Expr value=expression(0);require(";");return new TupleAssignmentStatement((TupleAccessExpr)left,value);
			}
			if (!(left instanceof VariableExpr))
				fail(current, "assignment requires a local variable or indexed local container");
			String operator = current.text;
			if (!(operator.equals("=") || operator.equals("+=") || operator.equals("-=")
					|| operator.equals("*=") || operator.equals("/=")))
				fail(current, "expected sampling or assignment operator");
			advance(); Expr value = expression(0); require(";");
			return new AssignmentStatement((VariableExpr) left, operator, value);
		}
		private Statement parseLocalDeclaration() {
			List<Expr> arrayShape = new ArrayList<Expr>();
			if (current.text.equals("array")) {
				advance(); require("["); arrayShape.addAll(dimensionList()); require("]");
			}
			Token typeToken = require(TokenKind.IDENTIFIER, "local declaration type expected");
			String type = typeToken.text;
			if (!isAssignmentType(type)) fail(typeToken, "unsupported local declaration type");
			if (accept("<")) {
				do { require(TokenKind.IDENTIFIER, "constraint name expected"); require("="); expression(4); }
				while (accept(","));
				require(">");
			}
			List<Expr> baseShape = new ArrayList<Expr>();
			if (accept("[")) { baseShape.addAll(dimensionList()); require("]"); }
			int expectedBase = type.equals("matrix") || type.equals("complex_matrix")
					|| type.equals("cholesky_factor_cov") ? 2
					: valueKind(type) == ValueKind.VECTOR || valueKind(type) == ValueKind.ROW_VECTOR
					|| valueKind(type) == ValueKind.COMPLEX_VECTOR
					|| valueKind(type) == ValueKind.COMPLEX_ROW_VECTOR
					|| type.equals("cov_matrix") || type.equals("corr_matrix")
					|| type.equals("cholesky_factor_corr") ? 1 : 0;
			if (baseShape.size() != expectedBase)
				fail(typeToken, type + " requires " + expectedBase + " dimension(s)");
			Token name = require(TokenKind.IDENTIFIER, "local variable name expected");
			if (accept("[")) { arrayShape.addAll(dimensionList()); require("]"); }
			List<Expr> shape = new ArrayList<Expr>(arrayShape); shape.addAll(baseShape);
			if (type.equals("cov_matrix") || type.equals("corr_matrix")
					|| type.equals("cholesky_factor_corr")) shape.add(baseShape.get(0));
			Expr initializer = accept("=") ? expression(0) : null; require(";");
			return new LocalDeclarationStatement(name.text, initializer, type.equals("int"),
					shape, valueKind(type), arrayShape.size());
		}
		private Statement parseTupleLocalDeclaration(){
			TupleTypeSpec type=parseTupleType();Token name=require(TokenKind.IDENTIFIER,"tuple variable name expected");
			Expr initializer=accept("=")?expression(0):null;require(";");
			return new TupleLocalDeclarationStatement(name.text,type,initializer);
		}
		private TupleTypeSpec parseTupleType(){
			require("tuple");require("(");List<TupleTypeSpec> members=new ArrayList<TupleTypeSpec>();
			members.add(parseTupleMemberType());
			if(!accept(","))fail(current,"tuple types require a comma-separated member list");
			if(current.text.equals(")")){advance();return new TupleTypeSpec(ValueKind.TUPLE,0,
					Collections.<Expr>emptyList(),members.toArray(new TupleTypeSpec[members.size()]));}
			do{members.add(parseTupleMemberType());}while(accept(","));require(")");
			return new TupleTypeSpec(ValueKind.TUPLE,0,Collections.<Expr>emptyList(),members.toArray(new TupleTypeSpec[members.size()]));
		}
		private TupleTypeSpec parseTupleMemberType(){
			if(current.text.equals("tuple"))return parseTupleType();
			List<Expr> arrayShape=new ArrayList<Expr>();
			if(current.text.equals("array")){advance();require("[");arrayShape.addAll(dimensionList());require("]");}
			Token type=require(TokenKind.IDENTIFIER,"tuple member type expected");
			if(!isAssignmentType(type.text))fail(type,"unsupported tuple member type");
			if(accept("<")){do{require(TokenKind.IDENTIFIER,"constraint name expected");require("=");expression(4);}while(accept(","));require(">");}
			List<Expr> baseShape=new ArrayList<Expr>();if(accept("[")){baseShape.addAll(dimensionList());require("]");}
			int expected=type.text.equals("matrix")||type.text.equals("complex_matrix")?2
					:valueKind(type.text)==ValueKind.VECTOR||valueKind(type.text)==ValueKind.ROW_VECTOR
					||valueKind(type.text)==ValueKind.COMPLEX_VECTOR||valueKind(type.text)==ValueKind.COMPLEX_ROW_VECTOR?1:0;
			if(baseShape.size()!=expected)fail(type,type.text+" requires "+expected+" dimension(s)");
			List<Expr> shape=new ArrayList<Expr>(arrayShape);shape.addAll(baseShape);
			return new TupleTypeSpec(valueKind(type.text),arrayShape.size(),shape,null);
		}
		private Expr expression(int minimumPrecedence) {
			Expr left;
			if (accept("+") || accept("-") || accept("!")) {
				String operator = previous; left = new UnaryExpr(operator, expression(7));
			} else if (accept("(")) {
				Expr first=expression(0);
				if(accept(",")){
					List<Expr> tuple=new ArrayList<Expr>();tuple.add(first);
					if(!current.text.equals(")")){tuple.add(expression(0));while(accept(","))tuple.add(expression(0));}
					require(")");left=new TupleExpr(tuple);
				}else{left=first;require(")");}
			}
			else if (accept("[")) { left = new LiteralExpr(literalElements("]"), false); require("]"); }
			else if (accept("{")) { left = new LiteralExpr(literalElements("}"), true); require("}"); }
			else if (current.kind == TokenKind.NUMBER) {
				Token number = current; advance();
				try {
					boolean imaginary = number.text.endsWith("i");
					String numeric = imaginary ? number.text.substring(0, number.text.length()-1) : number.text;
					left = new NumberExpr(Double.parseDouble(numeric), imaginary);
				}
				catch (NumberFormatException exception) { fail(number, "invalid number"); return null; }
			} else if (current.kind == TokenKind.IDENTIFIER) {
				Token identifier = current; advance();
				if (accept("(")) { List<Expr> args = arguments(); require(")"); left = new CallExpr(identifier.text, args); }
				else {
					List<IndexSpec> indices = new ArrayList<IndexSpec>();
					while (accept("[")) { indices.addAll(indexList()); require("]"); }
					left = new VariableExpr(identifier.text, indices);
				}
			} else { fail(current, "expression expected"); return null; }
			boolean postfix=true;while(postfix){
				if(accept("'"))left=new TransposeExpr(left);
				else if(current.kind==TokenKind.NUMBER&&current.text.matches("\\.[1-9][0-9]*")){
					int member=Integer.parseInt(current.text.substring(1));advance();left=new TupleAccessExpr(left,member-1);
				}else postfix=false;
			}
			while (true) {
				int precedence = precedence(current.text);
				if (precedence < minimumPrecedence) break;
				String operator = current.text; advance();
				left = new BinaryExpr(operator, left, expression(precedence + (operator.equals("^") ? 0 : 1)));
			}
			if (minimumPrecedence == 0 && accept("?")) {
				Expr whenTrue = expression(0); require(":"); Expr whenFalse = expression(0);
				left = new ConditionalExpr(left, whenTrue, whenFalse);
			}
			return left;
		}
		private List<Expr> literalElements(String closing) {
			List<Expr> result = new ArrayList<Expr>();
			if (current.text.equals(closing)) return result;
			result.add(expression(0));
			while (accept(",")) result.add(expression(0));
			return result;
		}
		private List<Expr> arguments() {
			List<Expr> result = new ArrayList<Expr>();
			if (current.text.equals(")")) return result;
			result.add(expression(0));
			while (accept(",") || accept("|")) result.add(expression(0));
			return result;
		}
		private List<IndexSpec> indexList() {
			List<IndexSpec> result = new ArrayList<IndexSpec>();
			do {
				if (accept(":")) result.add(IndexSpec.range(null, current.text.equals("]")
						|| current.text.equals(",") ? null : expression(0)));
				else {
					Expr lower = expression(0);
					if (accept(":")) result.add(IndexSpec.range(lower, current.text.equals("]")
							|| current.text.equals(",") ? null : expression(0)));
					else result.add(IndexSpec.single(lower));
				}
			} while (accept(","));
			return result;
		}
		private int precedence(String operator) {
			if (operator.equals("||")) return 0;
			if (operator.equals("&&")) return 1;
			if (operator.equals("==") || operator.equals("!=")) return 2;
			if (operator.equals("<") || operator.equals("<=")
					|| operator.equals(">") || operator.equals(">=")) return 3;
			if (operator.equals("+") || operator.equals("-")) return 4;
			if (operator.equals("*") || operator.equals("/")
					|| operator.equals(".*") || operator.equals("./")) return 5;
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
		for (Declaration declaration : declarations)
			result += elementCount(checkedShape(declaration, constants), declaration)
					* storageWidth(declaration.valueType().kind);
		return result;
	}
	private static int[] checkedShape(Declaration declaration, Context context) {
		int[] result = new int[declaration.shape.size()];
		for (int i = 0; i < result.length; i++)
			result[i] = checkedDimension(declaration.shape.get(i).eval(context).value, declaration);
		return result;
	}
	private static int elementCount(int[] shape, Declaration declaration) {
		long result = 1;
		for (int extent : shape) {
			result *= extent;
			if (result > 1000000L) throw new ModelScriptException(Collections.singletonList(
					declaration.error("container has too many elements")));
		}
		return (int) result;
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
	private static Map<String, int[]> defaultShapes(Map<String, double[]> data) {
		Map<String, int[]> result = new LinkedHashMap<String, int[]>();
		for (Map.Entry<String, double[]> entry : data.entrySet())
			result.put(entry.getKey(), entry.getValue().length == 1
					? new int[0] : new int[] {entry.getValue().length});
		return result;
	}
	private static Map<String, ValueType> defaultTypes(Map<String, double[]> data) {
		Map<String, ValueType> result = new LinkedHashMap<String, ValueType>();
		for (Map.Entry<String, double[]> entry : data.entrySet())
			result.put(entry.getKey(), new ValueType(ValueKind.SCALAR,
					entry.getValue().length == 1 ? 0 : 1));
		return result;
	}
	private static Map<String, ValueType> copyTypes(Map<String, ValueType> source) {
		return new LinkedHashMap<String, ValueType>(source);
	}
	private static ValueKind valueKind(String type) {
		if (type.equals("complex")) return ValueKind.COMPLEX;
		if (type.equals("complex_vector")) return ValueKind.COMPLEX_VECTOR;
		if (type.equals("complex_row_vector")) return ValueKind.COMPLEX_ROW_VECTOR;
		if (type.equals("complex_matrix")) return ValueKind.COMPLEX_MATRIX;
		if (type.equals("vector") || type.equals("simplex") || type.equals("ordered")
				|| type.equals("positive_ordered") || type.equals("sum_to_zero_vector")
				|| type.equals("unit_vector")) return ValueKind.VECTOR;
		if (type.equals("row_vector")) return ValueKind.ROW_VECTOR;
		if (type.equals("matrix") || type.equals("cov_matrix") || type.equals("corr_matrix")
				|| type.equals("cholesky_factor_cov") || type.equals("cholesky_factor_corr"))
			return ValueKind.MATRIX;
		return ValueKind.SCALAR;
	}
	private static boolean complexKind(ValueKind kind) {
		return kind == ValueKind.COMPLEX || kind == ValueKind.COMPLEX_VECTOR
				|| kind == ValueKind.COMPLEX_ROW_VECTOR || kind == ValueKind.COMPLEX_MATRIX;
	}
	private static int storageWidth(ValueKind kind) { return complexKind(kind) ? 2 : 1; }
	private static ValueKind complexKindFor(ValueKind kind) {
		if (kind == ValueKind.SCALAR) return ValueKind.COMPLEX;
		if (kind == ValueKind.VECTOR) return ValueKind.COMPLEX_VECTOR;
		if (kind == ValueKind.ROW_VECTOR) return ValueKind.COMPLEX_ROW_VECTOR;
		if (kind == ValueKind.MATRIX) return ValueKind.COMPLEX_MATRIX;
		return kind;
	}
	private static ValueKind realKindFor(ValueKind kind) {
		if (kind == ValueKind.COMPLEX) return ValueKind.SCALAR;
		if (kind == ValueKind.COMPLEX_VECTOR) return ValueKind.VECTOR;
		if (kind == ValueKind.COMPLEX_ROW_VECTOR) return ValueKind.ROW_VECTOR;
		if (kind == ValueKind.COMPLEX_MATRIX) return ValueKind.MATRIX;
		return kind;
	}
	private static RuntimeValue promoteComplex(RuntimeValue value, ValueKind target, Context context) {
		if (!complexKind(target) || complexKind(value.kind)) return value;
		if (complexKindFor(value.kind) != target)
			throw new IllegalArgumentException("cannot promote "+value.kind+" to "+target);
		Diff[] result = new Diff[value.values.length*2];
		for (int i = 0; i < value.values.length; i++) {
			result[2*i] = value.values[i]; result[2*i+1] = Diff.constant(0, context.dimension);
		}
		return new RuntimeValue(result, value.shape, target, value.arrayRank);
	}
	private static Diff[] complexApply(String operator, Diff ar, Diff ai, Diff br, Diff bi) {
		if(operator.equals("+")) return new Diff[]{ar.add(br),ai.add(bi)};
		if(operator.equals("-")) return new Diff[]{ar.subtract(br),ai.subtract(bi)};
		if(operator.equals("*")||operator.equals(".*"))
			return new Diff[]{ar.multiply(br).subtract(ai.multiply(bi)),ar.multiply(bi).add(ai.multiply(br))};
		if(operator.equals("/")||operator.equals("./")) {
			Diff denominator=br.multiply(br).add(bi.multiply(bi));
			return new Diff[]{ar.multiply(br).add(ai.multiply(bi)).divide(denominator),
					ai.multiply(br).subtract(ar.multiply(bi)).divide(denominator)};
		}
		throw new IllegalArgumentException("unsupported complex operator "+operator);
	}
	private static double[] identityMatrix(int dimension) {
		double[] result = new double[dimension * dimension];
		for (int i = 0; i < dimension; i++) result[i * dimension + i] = 1.0;
		return result;
	}
	private static Map<String, int[]> copyShapes(Map<String, int[]> source) {
		Map<String, int[]> result = new LinkedHashMap<String, int[]>();
		for (Map.Entry<String, int[]> entry : source.entrySet())
			result.put(entry.getKey(), entry.getValue().clone());
		return result;
	}
	private static double[] values(Diff[] source) {
		double[] result = new double[source.length];
		for (int i = 0; i < result.length; i++) result[i] = source[i].value;
		return result;
	}
}
