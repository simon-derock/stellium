import { StrictMode, Suspense, lazy } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// The operator page loads only at its own path, so the public bundle never carries it.
const OpsPage = lazy(() => import("./components/OpsPage").then((m) => ({ default: m.OpsPage })));
const isOps = window.location.pathname.replace(/\/$/, "") === "/ops";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {isOps ? (
      <Suspense fallback={null}>
        <OpsPage />
      </Suspense>
    ) : (
      <App />
    )}
  </StrictMode>,
);
