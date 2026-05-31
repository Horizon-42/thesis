import ReactDOM from "react-dom/client";
import { AppProvider } from "./context/AppContext";
import App from "./App";
// Import Cesium's default CSS for built-in widgets (timeline bar, animation wheel)
import "cesium/Build/Cesium/Widgets/widgets.css";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  // Cesium allocates WebGL resources and workers during mount. React StrictMode's
  // development-only remount doubles that cost, so the Cesium app shell is kept
  // outside StrictMode for realistic local performance profiling.
  <AppProvider>
    <App />
  </AppProvider>
);
