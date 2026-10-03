import { StrictMode, Suspense, lazy, type ComponentType } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

// Pages other than the console load only at their own path, so the console bundle never carries them.
const PAGES: Record<string, ComponentType> = {
  "/ops": lazy(() => import("./components/OpsPage").then((m) => ({ default: m.OpsPage }))),
  "/blog": lazy(() => import("./components/BlogPage").then((m) => ({ default: m.BlogPage }))),
};
const Page = PAGES[window.location.pathname.replace(/\/$/, "")];

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {Page ? (
      <Suspense fallback={null}>
        <Page />
      </Suspense>
    ) : (
      <App />
    )}
  </StrictMode>,
);
