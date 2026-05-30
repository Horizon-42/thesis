import * as Cesium from "cesium";
import type { AirportLocalTerrainState } from "../context/AppContext";
import {
  dsmTerrainTileRefsNearCoordinate,
  type DsmHeightmapTerrainMetadata,
  type DsmTerrainTileRef,
} from "./dsmHeightmapTerrain";

/**
 * Terrain runtime seam.
 *
 * Cesium terrain has several moving parts that must change together: provider
 * selection, tile streaming knobs, DSM warmup, no-imagery material, and lighting.
 * Keeping those rules here gives React hooks a narrow job: bind runtime policy
 * to component lifecycle without duplicating Cesium scene knowledge.
 */
export const AIRPORT_LOCAL_TERRAIN_SOURCE_LABEL = "Airport local DSM heightmap";

export const WORLD_TERRAIN_SETTINGS = {
  maximumScreenSpaceError: 1,
  minTileCacheSize: 512,
  loadingDescendantLimit: 20,
  preloadAncestors: true,
  preloadSiblings: true,
} as const;

export const LOCAL_TERRAIN_SETTINGS = {
  maximumScreenSpaceError: 0.5,
  minTileCacheSize: 256,
  focusPreloadConcurrency: 24,
  backgroundPreloadConcurrency: 8,
  focusMaxLevelTileRadius: 4,
  loadingDescendantLimit: 1000,
  preloadAncestors: true,
  preloadSiblings: true,
} as const;

const NO_IMAGERY_GLOBE_COLOR_CSS = "#8a8d84";
const NO_IMAGERY_CONTOUR_COLOR_CSS = "#4d514c";
const NO_IMAGERY_MAJOR_CONTOUR_COLOR_CSS = "#343832";
const NO_IMAGERY_LAMBERT_DIFFUSE_MULTIPLIER = 1.35;
const NO_IMAGERY_VERTEX_SHADOW_DARKNESS = 0.45;
const NO_IMAGERY_DEFAULT_MINIMUM_HEIGHT_METERS = -50;
const NO_IMAGERY_DEFAULT_MAXIMUM_HEIGHT_METERS = 350;
const NO_IMAGERY_MINIMUM_HEIGHT_SPAN_METERS = 10;
const NO_IMAGERY_CONTOUR_SPACING_METERS = 5;
const NO_IMAGERY_MAJOR_CONTOUR_SPACING_METERS = 25;

export const NO_IMAGERY_TERRAIN_MATERIAL_SOURCE = `
czm_material czm_getMaterial(czm_materialInput materialInput)
{
    czm_material material = czm_getDefaultMaterial(materialInput);
    float h = materialInput.height;
    float heightRange = max(maximumHeight - minimumHeight, 1.0);
    float heightMix = clamp((h - minimumHeight) / heightRange, 0.0, 1.0);
    float bandedHeight = floor(heightMix * elevationBandCount) / max(elevationBandCount, 1.0);
    float shade = mix(0.58, 1.22, heightMix) + (bandedHeight - 0.5) * 0.10;

#if (__VERSION__ == 300 || defined(GL_OES_standard_derivatives))
    float relief = clamp(length(vec2(dFdx(h), dFdy(h))) * reliefStrength, 0.0, 0.24);
    shade += relief;

    float fineCoordinate = h / contourSpacing;
    float fineDistance = min(fract(fineCoordinate), 1.0 - fract(fineCoordinate));
    float fineWidth = max(fwidth(fineCoordinate) * contourLineWidth, 0.012);
    float contour = 1.0 - smoothstep(0.0, fineWidth, fineDistance);

    float majorCoordinate = h / majorContourSpacing;
    float majorDistance = min(fract(majorCoordinate), 1.0 - fract(majorCoordinate));
    float majorWidth = max(fwidth(majorCoordinate) * majorContourLineWidth, 0.010);
    float majorContour = 1.0 - smoothstep(0.0, majorWidth, majorDistance);
#else
    float contourPhase = abs(fract(h / contourSpacing) - 0.5) * 2.0;
    float contour = 1.0 - smoothstep(0.0, 0.045, contourPhase);
    float majorContourPhase = abs(fract(h / majorContourSpacing) - 0.5) * 2.0;
    float majorContour = 1.0 - smoothstep(0.0, 0.040, majorContourPhase);
#endif

    vec3 color = baseColor.rgb * shade;
    color = mix(color, contourColor.rgb, contour * 0.32);
    color = mix(color, majorContourColor.rgb, majorContour * 0.58);

    material.diffuse = color;
    material.alpha = baseColor.a;
    return material;
}
`;

export interface GlobeTerrainStreamingSettings {
  maximumScreenSpaceError: number;
  tileCacheSize: number;
  loadingDescendantLimit: number;
  preloadAncestors: boolean;
  preloadSiblings: boolean;
  depthTestAgainstTerrain: boolean;
}

export interface GlobeRenderState {
  baseColor: Cesium.Color;
  enableLighting: boolean;
  showGroundAtmosphere: boolean;
  lambertDiffuseMultiplier: number;
  vertexShadowDarkness: number;
  material: Cesium.Material | undefined;
  viewerShadows: boolean;
  terrainShadows: Cesium.ShadowMode;
}

export interface HeightRange {
  minimumHeight: number;
  maximumHeight: number;
}

export interface TerrainHeightRange {
  minimumHeightM: number;
  maximumHeightM: number;
}

export interface LocalTerrainActivationPlan {
  heightRange: TerrainHeightRange;
  focusedTiles: DsmTerrainTileRef[];
  focusedPreloadConcurrency: number;
  backgroundPreloadConcurrency: number;
  activeTotalTiles: number;
}

export function captureTerrainStreamingSettings(
  globe: Cesium.Globe,
): GlobeTerrainStreamingSettings {
  return {
    maximumScreenSpaceError: globe.maximumScreenSpaceError,
    tileCacheSize: globe.tileCacheSize,
    loadingDescendantLimit: globe.loadingDescendantLimit,
    preloadAncestors: globe.preloadAncestors,
    preloadSiblings: globe.preloadSiblings,
    depthTestAgainstTerrain: globe.depthTestAgainstTerrain,
  };
}

export function applyWorldTerrainStreamingSettings(viewer: Cesium.Viewer): void {
  const { globe } = viewer.scene;

  globe.maximumScreenSpaceError = WORLD_TERRAIN_SETTINGS.maximumScreenSpaceError;
  globe.tileCacheSize = Math.max(globe.tileCacheSize, WORLD_TERRAIN_SETTINGS.minTileCacheSize);
  globe.loadingDescendantLimit = WORLD_TERRAIN_SETTINGS.loadingDescendantLimit;
  globe.preloadAncestors = WORLD_TERRAIN_SETTINGS.preloadAncestors;
  globe.preloadSiblings = WORLD_TERRAIN_SETTINGS.preloadSiblings;
  viewer.scene.requestRender();
}

export function applyLocalTerrainStreamingSettings(
  viewer: Cesium.Viewer,
  metadata: DsmHeightmapTerrainMetadata,
  maximumScreenSpaceError: number = LOCAL_TERRAIN_SETTINGS.maximumScreenSpaceError,
): void {
  const { globe } = viewer.scene;

  globe.maximumScreenSpaceError = maximumScreenSpaceError;
  globe.tileCacheSize = Math.max(
    globe.tileCacheSize,
    metadata.tileCount + 32,
    LOCAL_TERRAIN_SETTINGS.minTileCacheSize,
  );
  globe.preloadSiblings = LOCAL_TERRAIN_SETTINGS.preloadSiblings;
  globe.preloadAncestors = LOCAL_TERRAIN_SETTINGS.preloadAncestors;
  globe.loadingDescendantLimit = LOCAL_TERRAIN_SETTINGS.loadingDescendantLimit;
  globe.depthTestAgainstTerrain = true;
  viewer.scene.requestRender();
}

export function restoreTerrainStreamingSettings(
  viewer: Cesium.Viewer,
  settings: GlobeTerrainStreamingSettings,
): void {
  const { globe } = viewer.scene;

  globe.maximumScreenSpaceError = settings.maximumScreenSpaceError;
  globe.tileCacheSize = settings.tileCacheSize;
  globe.loadingDescendantLimit = settings.loadingDescendantLimit;
  globe.preloadAncestors = settings.preloadAncestors;
  globe.preloadSiblings = settings.preloadSiblings;
  globe.depthTestAgainstTerrain = settings.depthTestAgainstTerrain;
  viewer.scene.requestRender();
}

export function terrainHeightRange(
  metadata: DsmHeightmapTerrainMetadata,
): TerrainHeightRange {
  return {
    minimumHeightM: metadata.stats.min,
    maximumHeightM: metadata.stats.max,
  };
}

export function terrainFocusCoordinate(metadata: DsmHeightmapTerrainMetadata): {
  lon: number;
  lat: number;
} {
  return {
    lon: (metadata.bounds.west + metadata.bounds.east) / 2,
    lat: (metadata.bounds.south + metadata.bounds.north) / 2,
  };
}

export function focusedDsmTerrainTiles(
  metadata: DsmHeightmapTerrainMetadata,
): DsmTerrainTileRef[] {
  const focus = terrainFocusCoordinate(metadata);
  // Warm the max-detail tiles around the airport first so switching providers
  // does not expose the low-detail parent tile as a temporary flat patch.
  return dsmTerrainTileRefsNearCoordinate(metadata, focus.lon, focus.lat, {
    maxLevelRadius: LOCAL_TERRAIN_SETTINGS.focusMaxLevelTileRadius,
    ancestorRadius: 0,
  });
}

export function buildLocalTerrainActivationPlan(
  metadata: DsmHeightmapTerrainMetadata,
): LocalTerrainActivationPlan {
  return {
    heightRange: terrainHeightRange(metadata),
    focusedTiles: focusedDsmTerrainTiles(metadata),
    focusedPreloadConcurrency: LOCAL_TERRAIN_SETTINGS.focusPreloadConcurrency,
    backgroundPreloadConcurrency: LOCAL_TERRAIN_SETTINGS.backgroundPreloadConcurrency,
    activeTotalTiles: metadata.tileCount,
  };
}

export function airportLocalTerrainState(
  args: Omit<AirportLocalTerrainState, "sourceLabel"> & {
    sourceLabel?: string | null;
  },
): AirportLocalTerrainState {
  return {
    ...args,
    sourceLabel: args.sourceLabel ?? AIRPORT_LOCAL_TERRAIN_SOURCE_LABEL,
  };
}

export function airportLocalTerrainProgressState(args: {
  status: Extract<AirportLocalTerrainState["status"], "loading" | "preloading" | "active" | "error">;
  airportCode: string;
  heightRange?: TerrainHeightRange | null;
  loadedTiles?: number;
  totalTiles?: number;
  error?: string | null;
}): AirportLocalTerrainState {
  return airportLocalTerrainState({
    status: args.status,
    airportCode: args.airportCode,
    minimumHeightM: args.heightRange?.minimumHeightM ?? null,
    maximumHeightM: args.heightRange?.maximumHeightM ?? null,
    loadedTiles: args.loadedTiles ?? 0,
    totalTiles: args.totalTiles ?? 0,
    error: args.error ?? null,
  });
}

export function missingAirportLocalTerrainState(
  airportCode: string | null,
): AirportLocalTerrainState {
  return {
    status: "missing",
    airportCode,
    sourceLabel: null,
    minimumHeightM: null,
    maximumHeightM: null,
    loadedTiles: 0,
    totalTiles: 0,
    error: null,
  };
}

export function disabledAirportLocalTerrainState(
  airportCode: string | null,
): AirportLocalTerrainState {
  return {
    status: "disabled",
    airportCode,
    sourceLabel: null,
    minimumHeightM: null,
    maximumHeightM: null,
    loadedTiles: 0,
    totalTiles: 0,
    error: null,
  };
}

export function noImageryTerrainHeightRange(
  airportLocalTerrain: AirportLocalTerrainState,
): HeightRange {
  const { minimumHeightM, maximumHeightM } = airportLocalTerrain;
  if (
    minimumHeightM !== null &&
    maximumHeightM !== null &&
    Number.isFinite(minimumHeightM) &&
    Number.isFinite(maximumHeightM) &&
    maximumHeightM > minimumHeightM
  ) {
    const actualSpan = maximumHeightM - minimumHeightM;
    const targetSpan = Math.max(actualSpan, NO_IMAGERY_MINIMUM_HEIGHT_SPAN_METERS);
    const padding = (targetSpan - actualSpan) / 2;
    return {
      minimumHeight: minimumHeightM - padding,
      maximumHeight: maximumHeightM + padding,
    };
  }

  return {
    minimumHeight: NO_IMAGERY_DEFAULT_MINIMUM_HEIGHT_METERS,
    maximumHeight: NO_IMAGERY_DEFAULT_MAXIMUM_HEIGHT_METERS,
  };
}

export function captureGlobeRenderState(viewer: Cesium.Viewer): GlobeRenderState {
  const { globe } = viewer.scene;

  return {
    baseColor: Cesium.Color.clone(globe.baseColor),
    enableLighting: globe.enableLighting,
    showGroundAtmosphere: globe.showGroundAtmosphere,
    lambertDiffuseMultiplier: globe.lambertDiffuseMultiplier,
    vertexShadowDarkness: globe.vertexShadowDarkness,
    material: globe.material,
    viewerShadows: viewer.shadows,
    terrainShadows: viewer.terrainShadows,
  };
}

export function createNoImageryTerrainMaterial(): Cesium.Material {
  return new Cesium.Material({
    translucent: false,
    fabric: {
      uniforms: {
        baseColor: Cesium.Color.fromCssColorString(NO_IMAGERY_GLOBE_COLOR_CSS),
        contourColor: Cesium.Color.fromCssColorString(NO_IMAGERY_CONTOUR_COLOR_CSS),
        majorContourColor: Cesium.Color.fromCssColorString(NO_IMAGERY_MAJOR_CONTOUR_COLOR_CSS),
        minimumHeight: NO_IMAGERY_DEFAULT_MINIMUM_HEIGHT_METERS,
        maximumHeight: NO_IMAGERY_DEFAULT_MAXIMUM_HEIGHT_METERS,
        contourSpacing: NO_IMAGERY_CONTOUR_SPACING_METERS,
        majorContourSpacing: NO_IMAGERY_MAJOR_CONTOUR_SPACING_METERS,
        contourLineWidth: 1.4,
        majorContourLineWidth: 2.0,
        elevationBandCount: 18,
        reliefStrength: 0.045,
      },
      source: NO_IMAGERY_TERRAIN_MATERIAL_SOURCE,
    },
  });
}

export function configureNoImageryTerrainMaterial(
  material: Cesium.Material,
  heightRange: HeightRange,
): void {
  const uniforms = material.uniforms as Record<string, unknown>;
  uniforms.minimumHeight = heightRange.minimumHeight;
  uniforms.maximumHeight = heightRange.maximumHeight;
}

export function applyNoImageryTerrainStyle(
  viewer: Cesium.Viewer,
  material: Cesium.Material,
  heightRange: HeightRange,
): void {
  const { globe } = viewer.scene;

  configureNoImageryTerrainMaterial(material, heightRange);
  globe.baseColor = Cesium.Color.fromCssColorString(NO_IMAGERY_GLOBE_COLOR_CSS);
  globe.material = material;
  globe.showGroundAtmosphere = false;
  globe.enableLighting = true;
  globe.lambertDiffuseMultiplier = NO_IMAGERY_LAMBERT_DIFFUSE_MULTIPLIER;
  globe.vertexShadowDarkness = NO_IMAGERY_VERTEX_SHADOW_DARKNESS;
  viewer.shadows = true;
  viewer.terrainShadows = Cesium.ShadowMode.ENABLED;
  viewer.scene.requestRender();
}

export function restoreGlobeRenderState(
  viewer: Cesium.Viewer,
  state: GlobeRenderState,
): void {
  const { globe } = viewer.scene;
  const shouldRestoreAutoEnabledLighting = !state.enableLighting && globe.enableLighting;

  globe.baseColor = state.baseColor;
  if (shouldRestoreAutoEnabledLighting) {
    globe.enableLighting = state.enableLighting;
  }
  globe.showGroundAtmosphere = state.showGroundAtmosphere;
  globe.lambertDiffuseMultiplier = state.lambertDiffuseMultiplier;
  globe.vertexShadowDarkness = state.vertexShadowDarkness;
  globe.material = state.material;
  viewer.shadows = state.viewerShadows;
  viewer.terrainShadows = state.terrainShadows;
  viewer.scene.requestRender();
}
