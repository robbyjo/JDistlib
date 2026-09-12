/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Strict numeric file adapters for the model runner; arrays flatten in row-major order. */
final class ScriptDataFiles {
    private ScriptDataFiles() {}

    static String read(Path path) throws IOException {
        String text = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
        return text.startsWith("\ufeff") ? text.substring(1) : text;
    }

    static void put(Map<String, double[]> data, String name, double[] values) {
        if (!name.matches("[A-Za-z_][A-Za-z_0-9]*"))
            throw new IllegalArgumentException("invalid variable name: " + name);
        if (data.containsKey(name)) throw new IllegalArgumentException("duplicate data variable: " + name);
        data.put(name, values);
    }

    static void load(Map<String, double[]> data, String binding) throws IOException {
        int equals = binding.indexOf('=');
        if (equals < 0) {
            Path path = Paths.get(binding);
            try {
                for (Map.Entry<String, double[]> entry : new Json(read(path)).object().entrySet())
                    put(data, entry.getKey(), entry.getValue());
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException(path + ": " + exception.getMessage(), exception);
            }
        } else {
            String name = binding.substring(0, equals);
            Path path = Paths.get(binding.substring(equals + 1));
            try {
                String text = read(path);
                double[] values = path.toString().toLowerCase(Locale.ROOT).endsWith(".json")
                        ? new Json(text).numeric() : table(text, null);
                put(data, name, values);
            } catch (IllegalArgumentException exception) {
                throw new IllegalArgumentException(path + ": " + exception.getMessage(), exception);
            }
        }
    }

    static void column(Map<String, double[]> data, String binding) throws IOException {
        int equals = binding.indexOf('='), colon = binding.lastIndexOf(':');
        if (equals < 1 || colon <= equals + 1 || colon == binding.length() - 1)
            throw new IllegalArgumentException("--data-column requires variable=file.csv:header");
        Path path = Paths.get(binding.substring(equals + 1, colon));
        try {
            put(data, binding.substring(0, equals), table(read(path), binding.substring(colon + 1)));
        } catch (IllegalArgumentException exception) {
            throw new IllegalArgumentException(path + ": " + exception.getMessage(), exception);
        }
    }

    static void literal(Map<String, double[]> data, String binding) {
        int equals = binding.indexOf('=');
        if (equals < 1) throw new IllegalArgumentException("--set requires variable=number-or-JSON-array");
        put(data, binding.substring(0, equals), new Json(binding.substring(equals + 1)).numeric());
    }

    private static double[] table(String text, String header) {
        List<Double> values = new ArrayList<Double>();
        int width = -1, selected = -1, lineNumber = 0;
        String delimiter = null;
        for (String line : text.split("\\R", -1)) {
            lineNumber++;
            if (line.trim().isEmpty()) continue;
            if (delimiter == null) delimiter = line.indexOf('\t') >= 0 ? "\t" : ",";
            String[] cells = line.split(delimiter, -1);
            for (int i = 0; i < cells.length; i++) {
                cells[i] = cells[i].trim();
                if (cells[i].length() >= 2 && cells[i].startsWith("\"") && cells[i].endsWith("\""))
                    cells[i] = cells[i].substring(1, cells[i].length() - 1);
                if (cells[i].contains("\"")) throw new IllegalArgumentException("embedded CSV quotes are unsupported at line " + lineNumber);
            }
            if (width < 0) {
                width = cells.length;
                if (header != null) {
                    for (int i = 0; i < width; i++) if (cells[i].equals(header)) {
                        if (selected >= 0) throw new IllegalArgumentException("duplicate column: " + header);
                        selected = i;
                    }
                    if (selected < 0) throw new IllegalArgumentException("missing column: " + header);
                    continue;
                }
            }
            if (cells.length != width) throw new IllegalArgumentException("ragged table at line " + lineNumber);
            for (int i = 0; i < width; i++) if (header == null || i == selected) {
                try { values.add(new Json(cells[i]).scalar()); }
                catch (IllegalArgumentException exception) {
                    throw new IllegalArgumentException("expected finite number at line " + lineNumber + ", column " + (i + 1), exception);
                }
            }
        }
        if (header != null && width < 0) throw new IllegalArgumentException("missing column: " + header);
        return array(values);
    }

    private static double[] array(List<Double> values) {
        double[] result = new double[values.size()];
        for (int i = 0; i < result.length; i++) result[i] = values.get(i);
        return result;
    }

    /** Numeric JSON subset, with duplicate-key, rectangular-array and finite-number checks. */
    private static final class Json {
        private static final Pattern NUMBER = Pattern.compile("-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?");
        final String text;
        int position;
        Json(String text) { this.text = text; }
        Map<String, double[]> object() {
            Map<String, double[]> result = new LinkedHashMap<String, double[]>();
            require('{');
            if (!take('}')) {
                do {
                    require('"');
                    int start = position;
                    while (position < text.length() && text.charAt(position) != '"') position++;
                    String name = text.substring(start, position);
                    require('"'); require(':');
                    List<Double> values = new ArrayList<Double>();
                    value(values, 0);
                    put(result, name, array(values));
                } while (take(','));
                require('}');
            }
            end(); return result;
        }
        double[] numeric() {
            List<Double> values = new ArrayList<Double>();
            value(values, 0); end(); return array(values);
        }
        double scalar() {
            space(); Matcher matcher = NUMBER.matcher(text); matcher.region(position, text.length());
            if (!matcher.lookingAt()) throw error("expected a JSON number");
            double number = Double.parseDouble(matcher.group());
            position = matcher.end(); end();
            if (!Double.isFinite(number)) throw error("number must be finite");
            return number;
        }
        int[] value(List<Double> values, int depth) {
            if (depth > 64) throw error("array nesting exceeds 64");
            if (take('[')) {
                int count = 0; int[] child = null;
                if (!take(']')) {
                    do {
                        int[] shape = value(values, depth + 1);
                        if (child != null && !Arrays.equals(child, shape)) throw error("ragged JSON array");
                        child = shape; count++;
                    } while (take(','));
                    require(']');
                }
                int[] shape = new int[child == null ? 1 : child.length + 1];
                shape[0] = count;
                if (child != null) System.arraycopy(child, 0, shape, 1, child.length);
                return shape;
            }
            space(); Matcher matcher = NUMBER.matcher(text); matcher.region(position, text.length());
            if (!matcher.lookingAt()) throw error("expected a finite number or numeric array");
            double number = Double.parseDouble(matcher.group());
            if (!Double.isFinite(number)) throw error("number must be finite");
            position = matcher.end(); values.add(number); return new int[0];
        }
        void space() { while (position < text.length() && " \t\r\n".indexOf(text.charAt(position)) >= 0) position++; }
        boolean take(char token) {
            space(); if (position < text.length() && text.charAt(position) == token) { position++; return true; }
            return false;
        }
        void require(char token) { if (!take(token)) throw error("expected '" + token + "'"); }
        void end() { space(); if (position != text.length()) throw error("unexpected trailing input"); }
        IllegalArgumentException error(String message) {
            return new IllegalArgumentException(message + " at character " + (position + 1));
        }
    }
}
