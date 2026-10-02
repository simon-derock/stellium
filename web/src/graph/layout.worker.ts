// Runs the layout off the main thread, so settling the graph never freezes the intro.
import { settle, type LayoutRequest } from "./layout";

self.onmessage = (event: MessageEvent<LayoutRequest>) => {
  self.postMessage(settle(event.data));
};
