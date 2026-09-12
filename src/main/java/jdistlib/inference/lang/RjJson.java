/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.inference.lang;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Strict JSON reader for versioned RJ model-space documents. */
final class RjJson {
    private static final Pattern NUMBER = Pattern.compile("-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?");
    private final String text;
    private int position;
    private RjJson(String text) { this.text = text; }
    static Object parse(String text) {
        RjJson parser = new RjJson(text);
        Object result = parser.value(0);
        parser.space();
        if (parser.position != text.length()) throw parser.error("trailing JSON input");
        return result;
    }
    private Object value(int depth) {
        if (depth > 64) throw error("JSON nesting exceeds 64");
        space();
        if (take('{')) {
            Map<String, Object> result = new LinkedHashMap<String, Object>();
            if (!take('}')) {
                do {
                    String key = string(); require(':');
                    if (result.containsKey(key)) throw error("duplicate JSON key: " + key);
                    result.put(key, value(depth + 1));
                } while (take(','));
                require('}');
            }
            return result;
        }
        if (take('[')) {
            List<Object> result = new ArrayList<Object>();
            if (!take(']')) {
                do { result.add(value(depth + 1)); } while (take(','));
                require(']');
            }
            return result;
        }
        if (position < text.length() && text.charAt(position) == '"') return string();
        for (String literal : new String[] {"true", "false", "null"}) if (text.startsWith(literal, position)) {
            position += literal.length(); return "null".equals(literal) ? null : Boolean.valueOf(literal);
        }
        Matcher matcher = NUMBER.matcher(text); matcher.region(position, text.length());
        if (!matcher.lookingAt()) throw error("expected JSON value");
        double result = Double.parseDouble(matcher.group()); position = matcher.end();
        if (!Double.isFinite(result)) throw error("number must be finite");
        return Double.valueOf(result);
    }
    private String string() {
        require('"'); StringBuilder result = new StringBuilder();
        while (position < text.length()) {
            char c = text.charAt(position++);
            if (c == '"') return result.toString();
            if (c < 32) throw error("control character in string");
            if (c == '\\') {
                if (position == text.length()) throw error("unfinished escape");
                c = text.charAt(position++);
                switch (c) {
                case '"': case '\\': case '/': break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'u':
                    if (position + 4 > text.length()) throw error("unfinished unicode escape");
                    String hex = text.substring(position, position + 4);
                    if (!hex.matches("[0-9a-fA-F]{4}")) throw error("invalid unicode escape");
                    c = (char) Integer.parseInt(hex, 16); position += 4; break;
                default: throw error("invalid string escape");
                }
            }
            result.append(c);
        }
        throw error("unterminated string");
    }
    private void space() { while (position < text.length() && " \t\r\n".indexOf(text.charAt(position)) >= 0) position++; }
    private boolean take(char c) { space(); if (position < text.length() && text.charAt(position) == c) { position++; return true; } return false; }
    private void require(char c) { if (!take(c)) throw error("expected '" + c + "'"); }
    private IllegalArgumentException error(String message) { return new IllegalArgumentException(message + " at character " + (position + 1)); }
    static Map<String, Object> object(Object value, String label) {
        if (!(value instanceof Map<?, ?>)) throw new IllegalArgumentException(label + " must be an object");
        Map<String, Object> result = new LinkedHashMap<String, Object>();
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) result.put((String) entry.getKey(), entry.getValue());
        return result;
    }
    static List<?> list(Object value, String label) {
        if (!(value instanceof List<?>)) throw new IllegalArgumentException(label + " must be an array");
        return (List<?>) value;
    }
    static String string(Object value, String label) {
        if (!(value instanceof String) || ((String) value).trim().isEmpty()) throw new IllegalArgumentException(label + " must be a nonblank string");
        return (String) value;
    }
    static double number(Object value, String label) {
        if (!(value instanceof Double)) throw new IllegalArgumentException(label + " must be a finite number");
        return ((Double) value).doubleValue();
    }
    static void fields(Map<String, Object> object, String... allowed) {
        for (String key : object.keySet()) {
            boolean found = false;
            for (String field : allowed) found |= field.equals(key);
            if (!found) throw new IllegalArgumentException("unknown model-space field: " + key);
        }
        for (String field : allowed) if (!object.containsKey(field)) throw new IllegalArgumentException("missing model-space field: " + field);
    }
}
