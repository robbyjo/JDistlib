/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Concrete compute API or native library family. */
public enum ComputeApi { JAVA_CPU, ONEMKL, OPENBLAS, CUDA, OPENCL, VULKAN, AUTOMATIC }
