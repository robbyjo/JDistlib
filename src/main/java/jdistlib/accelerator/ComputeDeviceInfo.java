/* Copyright (C) 2026 Roby Joehanes; GPL-2.0-or-later */
package jdistlib.accelerator;

/** Immutable backend, runtime, driver, and device provenance. */
public final class ComputeDeviceInfo {
	private final String backendId, backendVersion, apiVersion, driverVersion;
	private final String vendor, device, architecture, deviceId;
	private final ComputeApi api;
	private final long globalMemoryBytes;

	public ComputeDeviceInfo(String backendId, String backendVersion, ComputeApi api,
			String apiVersion, String driverVersion, String vendor, String device,
			String architecture, String deviceId, long globalMemoryBytes) {
		if (backendId == null || backendVersion == null || api == null || apiVersion == null
				|| driverVersion == null || vendor == null || device == null
				|| architecture == null || deviceId == null || globalMemoryBytes < 0)
			throw new IllegalArgumentException("compute-device provenance fields are required");
		this.backendId = backendId; this.backendVersion = backendVersion; this.api = api;
		this.apiVersion = apiVersion; this.driverVersion = driverVersion; this.vendor = vendor;
		this.device = device; this.architecture = architecture; this.deviceId = deviceId;
		this.globalMemoryBytes = globalMemoryBytes;
	}
	public String backendId() { return backendId; }
	public String backendVersion() { return backendVersion; }
	public ComputeApi api() { return api; }
	public String apiVersion() { return apiVersion; }
	public String driverVersion() { return driverVersion; }
	public String vendor() { return vendor; }
	public String device() { return device; }
	public String architecture() { return architecture; }
	public String deviceId() { return deviceId; }
	public long globalMemoryBytes() { return globalMemoryBytes; }
	public String description() {
		return "backend=" + backendId + " " + backendVersion + ", api="
				+ api.name().toLowerCase(java.util.Locale.ROOT) + " " + apiVersion
				+ ", driver=" + driverVersion + ", device=" + vendor + " " + device
				+ ", architecture=" + architecture;
	}
}
