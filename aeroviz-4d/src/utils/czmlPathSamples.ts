/**
 * czmlPathSamples.ts
 * ------------------
 * The sampled positions of CZML packets, for framing the camera on them.
 */

/** A path point: degrees and metres above the ellipsoid (what a CZML `cartographicDegrees` holds). */
export interface PathSample {
  lon: number;
  lat: number;
  altM: number;
}

/**
 * The points of every packet whose id starts with one of `idPrefixes`. A packet's `position.cartographicDegrees` is
 * `[t, lon, lat, alt, t, lon, lat, alt, …]`; one without samples contributes none.
 */
export function czmlPathSamples(czml: unknown, idPrefixes: readonly string[]): PathSample[] {
  if (!Array.isArray(czml)) return [];
  const samples: PathSample[] = [];
  for (const raw of czml as unknown[]) {
    const packet = raw as { id?: unknown; position?: { cartographicDegrees?: unknown } };
    if (typeof packet.id !== "string" || !idPrefixes.some((prefix) => (packet.id as string).startsWith(prefix))) continue;
    const values = packet.position?.cartographicDegrees;
    if (!Array.isArray(values)) continue;
    for (let i = 0; i + 3 < values.length; i += 4) {
      samples.push({ lon: values[i + 1] as number, lat: values[i + 2] as number, altM: values[i + 3] as number });
    }
  }
  return samples;
}
