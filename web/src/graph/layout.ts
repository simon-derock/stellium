// Settles the force layout with the same forces the live graph uses. A few hundred ticks over
// ~330 nodes take a third of a second, so the page runs this in a worker (layout.worker.ts).
import { forceCenter, forceLink, forceManyBody, forceSimulation } from "d3-force-3d";
import { formForce, gravityForces, type ForceSettings } from "./forces";

export interface LayoutRequest {
  key: string;
  nodes: { id: string; x: number; y: number }[];
  links: { source: string; target: string }[];
  targets: [string, { x: number; y: number }][];
  settings: ForceSettings;
}

export interface LayoutResult {
  key: string;
  positions: [string, { x: number; y: number }][];
}

const TICKS = 300;

export function settle({ key, nodes, links, targets, settings }: LayoutRequest): LayoutResult {
  const bodies = nodes.map((node) => ({ ...node }));
  const [gx, gy] = gravityForces(settings.gravity);
  const simulation = forceSimulation(bodies, 2)
    .stop()
    .force("charge", forceManyBody().strength(settings.charge))
    .force(
      "link",
      forceLink(links.map((link) => ({ ...link })))
        .id((node) => node.id)
        .distance(settings.linkDistance)
        .strength(settings.linkStrength),
    )
    .force("center", forceCenter().strength(settings.center))
    .force("x", gx)
    .force("y", gy)
    .force("form", formForce(new Map(targets), settings.form))
    .velocityDecay(0.62)
    // alpha falls from 1 to d3's resting threshold over exactly TICKS steps
    .alphaDecay(1 - Math.pow(0.001, 1 / TICKS));
  for (let i = 0; i < TICKS; i += 1) simulation.tick();
  return { key, positions: bodies.map((body) => [body.id, { x: body.x, y: body.y }]) };
}
