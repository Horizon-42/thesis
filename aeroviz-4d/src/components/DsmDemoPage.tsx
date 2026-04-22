import { useEffect, useRef, useState } from "react";
import * as Cesium from "cesium";
import { useApp } from "../context/AppContext";
import { airportDsm3DTilesUrl } from "../data/airportData";

interface DsmDemoState {
  status: string;
  rasterSize: string;
  meshSize: string;
  triangles: string;
  renderer: string;
  location: string;
  overlay: string;
}

export default function DsmDemoPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { activeAirportCode } = useApp();
  const [state, setState] = useState<DsmDemoState>({
    status: "Loading DSM 3D Tiles",
    rasterSize: "",
    meshSize: "",
    triangles: "",
    renderer: "3D Tiles",
    location: "",
    overlay: "",
  });

  useEffect(() => {
    if (!containerRef.current || !activeAirportCode) return;

    const metadataUrl = airportDsm3DTilesUrl(activeAirportCode, "metadata.json");
    const tilesetUrl = airportDsm3DTilesUrl(activeAirportCode, "tileset.json");
    const glbUrl = airportDsm3DTilesUrl(activeAirportCode, "dsm.glb");

    let cancelled = false;
    let fallbackTimer: number | undefined;
    let tileWasVisible = false;
    const viewer = new Cesium.Viewer(containerRef.current, {
      baseLayer: false,
      baseLayerPicker: false,
      geocoder: false,
      homeButton: false,
      sceneModePicker: false,
      navigationHelpButton: false,
      animation: false,
      timeline: false,
      infoBox: false,
      selectionIndicator: false,
      fullscreenButton: false,
      skyAtmosphere: new Cesium.SkyAtmosphere(),
    });

    viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#020617");
    viewer.scene.highDynamicRange = false;
    viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;
    viewer.scene.globe.depthTestAgainstTerrain = false;
    viewer.scene.globe.baseColor = Cesium.Color.fromCssColorString("#1f4f46");

    Cesium.TileMapServiceImageryProvider.fromUrl(
      Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII")
    )
      .then((provider) => {
        if (cancelled || viewer.isDestroyed()) return;
        viewer.imageryLayers.addImageryProvider(provider, 0);
        viewer.scene.requestRender();
      })
      .catch((error) => {
        console.warn("[DsmDemoPage] Natural Earth imagery failed:", error);
      });

    fetch(metadataUrl)
      .then((response) => response.json())
      .then(async (metadata) => {
        if (cancelled || viewer.isDestroyed()) return;

        const footprint = [
          metadata.corners.northWest,
          metadata.corners.northEast,
          metadata.corners.southEast,
          metadata.corners.southWest,
          metadata.corners.northWest,
        ].flatMap(({ lon, lat }) => [lon, lat]);
        viewer.entities.add({
          name: "GeoTIFF footprint",
          polyline: {
            positions: Cesium.Cartesian3.fromDegreesArray(footprint),
            width: 3,
            material: Cesium.Color.RED,
            clampToGround: true,
          },
        });

        const modelMatrix = Cesium.Transforms.eastNorthUpToFixedFrame(
          Cesium.Cartesian3.fromDegrees(metadata.center.lon, metadata.center.lat, 0)
        );
        if (metadata.exactOverlay?.url) {
          const exactOverlayModel = await Cesium.Model.fromGltfAsync({
            url: metadata.exactOverlay.url,
            modelMatrix,
            upAxis: Cesium.Axis.Z,
            forwardAxis: Cesium.Axis.X,
            backFaceCulling: false,
            cull: false,
          });
          if (cancelled || viewer.isDestroyed()) {
            exactOverlayModel.destroy();
            return;
          }
          viewer.scene.primitives.add(exactOverlayModel);
        }

        const modelHeight = (metadata.stats.max - metadata.stats.min) * metadata.verticalExaggeration;
        const tileset = await Cesium.Cesium3DTileset.fromUrl(tilesetUrl, {
          modelUpAxis: Cesium.Axis.Z,
          modelForwardAxis: Cesium.Axis.X,
          backFaceCulling: false,
          maximumScreenSpaceError: 1,
        });
        if (cancelled || viewer.isDestroyed()) {
          tileset.destroy();
          return;
        }
        tileset.tileFailed.addEventListener((error) => {
          console.error("[DsmDemoPage] DSM tile failed:", error);
          setState((current) => ({ ...current, status: "DSM tile failed to render" }));
        });
        viewer.scene.primitives.add(tileset);
        tileset.tileVisible.addEventListener(() => {
          tileWasVisible = true;
        });
        tileset.allTilesLoaded.addEventListener(() => {
          setState((current) => ({ ...current, status: "DSM 3D Tiles loaded" }));
        });

        fallbackTimer = window.setTimeout(async () => {
          if (cancelled || viewer.isDestroyed()) return;
          if (tileWasVisible) return;

          console.warn("[DsmDemoPage] 3D Tiles produced no draw commands; showing direct GLB fallback.");
          const model = await Cesium.Model.fromGltfAsync({
            url: glbUrl,
            modelMatrix,
            upAxis: Cesium.Axis.Z,
            forwardAxis: Cesium.Axis.X,
            backFaceCulling: false,
            cull: false,
            color: Cesium.Color.fromCssColorString("#74f08b"),
            colorBlendMode: Cesium.ColorBlendMode.MIX,
            colorBlendAmount: 0.35,
          });
          if (cancelled || viewer.isDestroyed()) {
            model.destroy();
            return;
          }
          viewer.scene.primitives.add(model);
          viewer.scene.requestRender();
          setState((current) => ({
            ...current,
            status: "DSM direct GLB fallback shown",
            renderer: "3D Tiles generated; direct GLB fallback visible",
          }));
        }, 2500);

        const cameraOffset = new Cesium.HeadingPitchRange(
          Cesium.Math.toRadians(-35),
          Cesium.Math.toRadians(-42),
          5200
        );
        const focus = Cesium.Cartesian3.fromDegrees(
          metadata.center.lon,
          metadata.center.lat,
          metadata.stats.min + modelHeight / 2
        );
        viewer.camera.lookAt(focus, cameraOffset);
        viewer.camera.lookAtTransform(Cesium.Matrix4.IDENTITY);
        viewer.scene.requestRender();

        setState({
          status: "DSM model placed on globe",
          rasterSize: metadata?.raster
            ? `${metadata.raster.width} x ${metadata.raster.height}`
            : "Unknown",
          meshSize: metadata?.vertices ? `${metadata.vertices.toLocaleString()} vertices` : "Unknown",
          triangles: metadata?.triangles ? `${metadata.triangles.toLocaleString()}` : "Unknown",
          renderer: "3D Tiles on Cesium globe",
          location: `${metadata.center.lat.toFixed(6)}, ${metadata.center.lon.toFixed(6)}`,
          overlay: metadata.exactOverlay?.url ? "Exact projected overlay model" : "Footprint only",
        });
      })
      .catch((error) => {
        if (cancelled) return;
        console.error("[DsmDemoPage] Failed to load DSM demo:", error);
        setState({
          status: "DSM demo load failed",
          rasterSize: "",
          meshSize: "",
          triangles: "",
          renderer: "Failed",
          location: "",
          overlay: "",
        });
      });

    return () => {
      cancelled = true;
      if (fallbackTimer !== undefined) window.clearTimeout(fallbackTimer);
      viewer.destroy();
    };
  }, [activeAirportCode]);

  return (
    <main className="dsm-demo-page">
      <div ref={containerRef} className="dsm-demo-viewer" />
      <section className="dsm-demo-panel">
        <a href="/" className="dsm-demo-link">Flight view</a>
        <h1>{activeAirportCode || "Airport"} DSM</h1>
        <p>{state.status}</p>
        <dl>
          <div>
            <dt>Raster</dt>
            <dd>{state.rasterSize || "Pending"}</dd>
          </div>
          <div>
            <dt>Mesh</dt>
            <dd>{state.meshSize || "Pending"}</dd>
          </div>
          <div>
            <dt>Triangles</dt>
            <dd>{state.triangles || "Pending"}</dd>
          </div>
          <div>
            <dt>Source</dt>
            <dd>{state.renderer}</dd>
          </div>
          <div>
            <dt>Center</dt>
            <dd>{state.location || "Pending"}</dd>
          </div>
          <div>
            <dt>Overlay</dt>
            <dd>{state.overlay || "Pending"}</dd>
          </div>
        </dl>
      </section>
    </main>
  );
}
