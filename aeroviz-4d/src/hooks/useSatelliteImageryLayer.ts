import { useEffect, useRef } from "react";
import * as Cesium from "cesium";
import { useApp, type AirportLocalTerrainState } from "../context/AppContext";

const NO_IMAGERY_GLOBE_COLOR = Cesium.Color.fromCssColorString("#8a8d84");
const NO_IMAGERY_CONTOUR_COLOR = Cesium.Color.fromCssColorString("#4d514c");
const NO_IMAGERY_MAJOR_CONTOUR_COLOR = Cesium.Color.fromCssColorString("#343832");
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

interface GlobeRenderState {
  baseColor: Cesium.Color;
  enableLighting: boolean;
  showGroundAtmosphere: boolean;
  lambertDiffuseMultiplier: number;
  vertexShadowDarkness: number;
  material: Cesium.Material | undefined;
  viewerShadows: boolean;
  terrainShadows: Cesium.ShadowMode;
}

interface HeightRange {
  minimumHeight: number;
  maximumHeight: number;
}

function requestRender(viewer: Cesium.Viewer): void {
  viewer.scene.requestRender();
}

function captureGlobeRenderState(viewer: Cesium.Viewer): GlobeRenderState {
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

function createNoImageryTerrainMaterial(): Cesium.Material {
  return new Cesium.Material({
    translucent: false,
    fabric: {
      uniforms: {
        baseColor: NO_IMAGERY_GLOBE_COLOR,
        contourColor: NO_IMAGERY_CONTOUR_COLOR,
        majorContourColor: NO_IMAGERY_MAJOR_CONTOUR_COLOR,
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

function configureNoImageryTerrainMaterial(
  material: Cesium.Material,
  heightRange: HeightRange,
): void {
  const uniforms = material.uniforms as Record<string, unknown>;
  uniforms.minimumHeight = heightRange.minimumHeight;
  uniforms.maximumHeight = heightRange.maximumHeight;
}

function applyNoImageryTerrainStyle(
  viewer: Cesium.Viewer,
  material: Cesium.Material,
  heightRange: HeightRange,
): void {
  const { globe } = viewer.scene;

  configureNoImageryTerrainMaterial(material, heightRange);
  globe.baseColor = Cesium.Color.clone(NO_IMAGERY_GLOBE_COLOR);
  globe.material = material;
  globe.showGroundAtmosphere = false;
  globe.enableLighting = true;
  globe.lambertDiffuseMultiplier = NO_IMAGERY_LAMBERT_DIFFUSE_MULTIPLIER;
  globe.vertexShadowDarkness = NO_IMAGERY_VERTEX_SHADOW_DARKNESS;
  viewer.shadows = true;
  viewer.terrainShadows = Cesium.ShadowMode.ENABLED;
  requestRender(viewer);
}

function restoreGlobeRenderState(viewer: Cesium.Viewer, state: GlobeRenderState): void {
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
  requestRender(viewer);
}

/**
 * Toggle the Cesium base imagery layer.  The Viewer creates this layer during
 * initialization, so this hook syncs visibility and provides a gray shaded
 * terrain style for the no-imagery state.
 */
export function useSatelliteImageryLayer(): void {
  const { viewer, layers, airportLocalTerrain } = useApp();
  const previousGlobeStateRef = useRef<GlobeRenderState | null>(null);
  const noImageryMaterialRef = useRef<Cesium.Material | null>(null);

  useEffect(() => {
    if (!viewer || viewer.isDestroyed()) return;

    if (viewer.imageryLayers.length === 0) return;

    const baseLayer = viewer.imageryLayers.get(0);
    if (!baseLayer) return;

    baseLayer.show = layers.satelliteImagery;
    if (layers.satelliteImagery) {
      if (previousGlobeStateRef.current) {
        restoreGlobeRenderState(viewer, previousGlobeStateRef.current);
        previousGlobeStateRef.current = null;
        if (noImageryMaterialRef.current && !noImageryMaterialRef.current.isDestroyed()) {
          noImageryMaterialRef.current.destroy();
        }
        noImageryMaterialRef.current = null;
      } else {
        requestRender(viewer);
      }
      return;
    }

    if (!previousGlobeStateRef.current) {
      previousGlobeStateRef.current = captureGlobeRenderState(viewer);
    }
    if (!noImageryMaterialRef.current || noImageryMaterialRef.current.isDestroyed()) {
      noImageryMaterialRef.current = createNoImageryTerrainMaterial();
    }
    applyNoImageryTerrainStyle(
      viewer,
      noImageryMaterialRef.current,
      noImageryTerrainHeightRange(airportLocalTerrain),
    );
  }, [
    viewer,
    layers.satelliteImagery,
    airportLocalTerrain.minimumHeightM,
    airportLocalTerrain.maximumHeightM,
  ]);

  useEffect(() => {
    return () => {
      if (!viewer || viewer.isDestroyed() || !previousGlobeStateRef.current) return;

      restoreGlobeRenderState(viewer, previousGlobeStateRef.current);
      previousGlobeStateRef.current = null;
      if (noImageryMaterialRef.current && !noImageryMaterialRef.current.isDestroyed()) {
        noImageryMaterialRef.current.destroy();
      }
      noImageryMaterialRef.current = null;
    };
  }, [viewer]);
}
