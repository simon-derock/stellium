// One definition of the physics, shared by the live graph and the layout worker. The worker
// settles a layout with exactly these forces, so when the live simulation takes over the nodes are
// already at rest and nothing drifts.
import { forceX, forceY, type Force } from "d3-force-3d";
import type { VizProfile } from "./graph";

export interface ForceSettings {
  charge: number;
  linkDistance: number;
  linkStrength: number;
  center: number;
  gravity: number;
  form: number;
}

export function forceSettings(viz: VizProfile): ForceSettings {
  const formed = viz.formStrength > 0;
  return {
    charge: formed ? viz.charge * 0.06 : viz.charge,
    linkDistance: viz.linkDistance,
    linkStrength: formed ? 0.01 : 1,
    center: formed ? 0 : 1,
    // A faint pull to the middle keeps small islands of a free layout from drifting out of frame.
    gravity: formed ? 0 : 0.035,
    form: viz.formStrength,
  };
}

type Body = { id: string; x: number; y: number; vx: number; vy: number; fx?: number; fy?: number };

// Spring each node toward its place in the formation, independent of the cooling alpha so the
// silhouette still holds once the simulation relaxes. Pinned nodes belong to the pointer or intro.
export function formForce(targets: Map<string, { x: number; y: number }>, strength: number): Force {
  let bodies: Body[] = [];
  const force = (() => {
    if (!strength) return;
    for (const body of bodies) {
      if (body.fx != null || body.fy != null) continue;
      const target = targets.get(body.id);
      if (!target) continue;
      body.vx += (target.x - body.x) * strength;
      body.vy += (target.y - body.y) * strength;
    }
  }) as Force;
  force.initialize = (nodes) => {
    bodies = nodes as Body[];
  };
  return force;
}

export function gravityForces(gravity: number): [Force, Force] {
  return [forceX(0).strength(gravity), forceY(0).strength(gravity)];
}
