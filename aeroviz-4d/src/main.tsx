import React from "react";
import ReactDOM from "react-dom/client";
import { AppProvider } from "./context/AppContext";
import App from "./App";
// Import Cesium's default CSS for built-in widgets (timeline bar, animation wheel)
import "cesium/Build/Cesium/Widgets/widgets.css";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  // StrictMode renders components twice in development to catch side-effects.
  // NOTE: This causes the CesiumJS viewer to be created TWICE in dev.
  // If you see a duplicate globe, this is why — it's intentional.
  <React.StrictMode>
    <AppProvider>
      <App />
    </AppProvider>
  </React.StrictMode>
);
