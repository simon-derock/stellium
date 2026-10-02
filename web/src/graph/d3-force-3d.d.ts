// Just the slice of d3-force-3d the layout uses; the package ships no types of its own.
declare module "d3-force-3d" {
  export type Force = ((alpha: number) => void) & { initialize?: (nodes: unknown[]) => void };
  interface Strength<T> {
    strength(value: number): T;
  }
  export interface ManyBody extends Force, Strength<ManyBody> {}
  export interface Center extends Force, Strength<Center> {}
  export interface Axis extends Force, Strength<Axis> {}
  export interface Link extends Force, Strength<Link> {
    id(accessor: (node: { id: string }) => string): Link;
    distance(value: number): Link;
  }
  export interface Simulation {
    force(name: string, force: Force | null): Simulation;
    stop(): Simulation;
    tick(): Simulation;
    alphaDecay(value: number): Simulation;
    velocityDecay(value: number): Simulation;
  }
  export function forceSimulation(nodes: unknown[], dimensions?: number): Simulation;
  export function forceManyBody(): ManyBody;
  export function forceCenter(): Center;
  export function forceLink(links?: unknown[]): Link;
  export function forceX(x?: number): Axis;
  export function forceY(y?: number): Axis;
}
