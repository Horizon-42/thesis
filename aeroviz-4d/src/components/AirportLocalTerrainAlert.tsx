import { useEffect, useMemo, useState } from "react";
import { useApp } from "../context/AppContext";

function terrainAlertText(args: {
  status: "missing" | "error";
  airportCode: string | null;
  error: string | null;
}): { title: string; body: string; detail?: string } {
  const airportCode = args.airportCode ?? "the active airport";
  if (args.status === "missing") {
    return {
      title: "Local terrain data is missing",
      body:
        `No airport-local terrain package was found for ${airportCode}. ` +
        `Generate public/data/airports/${airportCode}/dsm/heightmap-terrain/metadata.json, ` +
        "then reload or switch airports.",
    };
  }

  return {
    title: "Local terrain data needs attention",
    body:
      `The airport-local terrain package for ${airportCode} could not be loaded. ` +
      "If this package was generated before precision metadata was added, regenerate it so the app can choose sources by resolution.",
    detail: args.error ?? undefined,
  };
}

export default function AirportLocalTerrainAlert() {
  const { airportLocalTerrain, layers } = useApp();
  const [dismissedKey, setDismissedKey] = useState<string | null>(null);
  const shouldShow =
    layers.airportLocalTerrain &&
    (airportLocalTerrain.status === "missing" || airportLocalTerrain.status === "error");
  const alertKey = shouldShow
    ? [
        airportLocalTerrain.airportCode ?? "unknown-airport",
        airportLocalTerrain.status,
        airportLocalTerrain.error ?? "",
      ].join(":")
    : null;

  useEffect(() => {
    if (alertKey && alertKey !== dismissedKey) {
      setDismissedKey(null);
    }
  }, [alertKey, dismissedKey]);

  const alertText = useMemo(() => {
    if (!shouldShow) return null;
    if (
      airportLocalTerrain.status !== "missing" &&
      airportLocalTerrain.status !== "error"
    ) {
      return null;
    }

    return terrainAlertText({
      status: airportLocalTerrain.status,
      airportCode: airportLocalTerrain.airportCode,
      error: airportLocalTerrain.error,
    });
  }, [
    shouldShow,
    airportLocalTerrain.status,
    airportLocalTerrain.airportCode,
    airportLocalTerrain.error,
  ]);

  if (!alertText || !alertKey || dismissedKey === alertKey) return null;

  return (
    <div className="airport-local-terrain-alert-backdrop">
      <section
        className="airport-local-terrain-alert"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="airport-local-terrain-alert-title"
      >
        <h2 id="airport-local-terrain-alert-title">{alertText.title}</h2>
        <p>{alertText.body}</p>
        {alertText.detail ? <pre>{alertText.detail}</pre> : null}
        <button type="button" onClick={() => setDismissedKey(alertKey)}>
          Dismiss
        </button>
      </section>
    </div>
  );
}
