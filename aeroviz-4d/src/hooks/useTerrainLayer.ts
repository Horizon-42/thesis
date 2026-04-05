import { useEffect } from "react";
import { useApp } from "../context/AppContext";

export function useTerrainLayer(): void {
  const { viewer, layers } = useApp();

  useEffect(() => {
    if (!viewer) return;
    viewer.scene.globe.show = layers.terrain;
  }, [viewer, layers.terrain]);
}
