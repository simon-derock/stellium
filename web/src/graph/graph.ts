/* ------------------------------------------------------------------ *
 * STELLIUM graph frame: DTOs from /api/v1/graph/snapshot, plus the
 * themes, draw styles and formations ported from the Lunarbit canvas.
 * ------------------------------------------------------------------ */

/** Six colour slots; each names a node kind in the Olympic graph. */
export type LayerId = "evidence" | "games" | "event" | "sport" | "venue" | "answer";

export interface GraphNode {
  id: string;
  type: string;
  layer: LayerId;
  label: string;
  weight: number;
  source_count: number;
  confidence: number;
  privacy_state: "public" | "redacted";
  scope: string;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relationship_type: string;
  confidence: number;
  provenance_label: string;
}

export interface Snapshot {
  graph_nodes: GraphNode[];
  graph_edges: GraphEdge[];
  disclosure: string;
}

export const LAYER_NAMES: Record<LayerId, string> = {
  evidence: "Cited event",
  games: "Games",
  event: "Event",
  sport: "Sport",
  venue: "Venue",
  answer: "Answer",
};

/* ------------------------------------------------------------------ *
 * Views: which projection of the graph is drawn
 * ------------------------------------------------------------------ */

export type ViewId = "constellation" | "venues" | "lineage" | "investigation";

export interface ViewProfile {
  id: ViewId;
  name: string;
  hint: string;
}

export const VIEWS: ViewProfile[] = [
  { id: "investigation", name: "Investigation", hint: "The events a question touched · editions · rivals" },
  { id: "constellation", name: "Constellation", hint: "Every Games joined to the sports it held" },
  { id: "venues", name: "Venues", hint: "Venues linked to the Games they hosted" },
  { id: "lineage", name: "Lineage", hint: "One sport, each event chained to its previous edition" },
];

/* ------------------------------------------------------------------ *
 * Themes: each preset is a distinct ink/paper gamut, not a filter.
 * ------------------------------------------------------------------ */

export interface Palette {
  paper: string;
  ink: string;
  edge: string;
  edgeHot: string;
  layers: Record<LayerId, string>;
  /** chromatic ramp used by multi-colour draw modes */
  chroma: string[];
}

export interface ThemePreset {
  id: string;
  name: string;
  hint: string;
  vars: Record<string, string>;
  palette: Palette;
}

export const THEMES: ThemePreset[] = [
  {
    id: "obsidian-observatory",
    name: "Obsidian Observatory",
    hint: "Mineral black · platinum orbit · champagne signal",
    vars: {
      "--background": "#0a0b0f",
      "--surface": "#12151b",
      "--foreground": "#edf0f2",
      "--muted-foreground": "#8d969f",
      "--border": "#27303a",
      "--primary": "#d7dde2",
      "--primary-foreground": "#0a0b0f",
      "--accent": "#d1ad72",
    },
    palette: {
      paper: "#0a0b0f",
      ink: "#edf0f2",
      edge: "rgba(215,221,226,0.18)",
      edgeHot: "rgba(255,244,215,0.92)",
      layers: {
        evidence: "#edf0f2",
        games: "#d1ad72",
        event: "#b8c3cc",
        sport: "#8b9ba8",
        venue: "#f1d39a",
        answer: "#65717c",
      },
      chroma: ["#edf0f2", "#f1d39a", "#d1ad72", "#b8c3cc", "#8b9ba8", "#65717c"],
    },
  },
  {
    id: "carbon",
    name: "Carbon",
    hint: "Ink-blue vellum · jewel signal",
    vars: {
      "--background": "#07080c",
      "--surface": "#0d0f15",
      "--foreground": "#e8ebf2",
      "--muted-foreground": "#8a93a4",
      "--border": "#1b1f29",
      "--primary": "#e8ebf2",
      "--primary-foreground": "#07080c",
      "--accent": "#cba36a",
    },
    palette: {
      paper: "#07080c",
      ink: "#e8ebf2",
      edge: "rgba(160,178,206,0.13)",
      edgeHot: "rgba(226,236,255,0.78)",
      layers: {
        evidence: "#eef2fa",
        games: "#cba36a",
        event: "#7fb3a6",
        sport: "#8f9ec4",
        venue: "#e7cf9a",
        answer: "#6a7490",
      },
      chroma: ["#eef2fa", "#cba36a", "#e0a37e", "#7fb3a6", "#8f9ec4", "#a98fb8"],
    },
  },
  {
    id: "aurum-dust",
    name: "Aurum Dust",
    hint: "Obsidian velvet · floating champagne gold",
    vars: {
      "--background": "#12100d",
      "--surface": "#1d1812",
      "--foreground": "#f8f0dc",
      "--muted-foreground": "#b7a888",
      "--border": "#6b4a18",
      "--primary": "#e6bd67",
      "--primary-foreground": "#12100d",
      "--accent": "#f4d58b",
    },
    palette: {
      paper: "#12100d",
      ink: "#f8f0dc",
      edge: "rgba(214,171,86,0.34)",
      edgeHot: "rgba(255,239,184,0.98)",
      layers: {
        evidence: "#fff4cf",
        games: "#e6bd67",
        event: "#f4d58b",
        sport: "#b99043",
        venue: "#ffe8a8",
        answer: "#81652e",
      },
      chroma: ["#fff4cf", "#ffe8a8", "#f4d58b", "#e6bd67", "#b99043", "#81652e"],
    },
  },
  {
    id: "glass-observatory",
    name: "Glass Observatory",
    hint: "Blue-black glass · ice light · violet depth",
    vars: {
      "--background": "#080d16",
      "--surface": "#101a2a",
      "--foreground": "#e9f4ff",
      "--muted-foreground": "#91a6bd",
      "--border": "#263b55",
      "--primary": "#b7ddff",
      "--primary-foreground": "#080d16",
      "--accent": "#b9a8ff",
    },
    palette: {
      paper: "#080d16",
      ink: "#e9f4ff",
      edge: "rgba(151,207,255,0.18)",
      edgeHot: "rgba(225,242,255,0.98)",
      layers: { evidence: "#e9f4ff", games: "#b7ddff", event: "#b9a8ff", sport: "#6da4d1", venue: "#d8cfff", answer: "#526f98" },
      chroma: ["#e9f4ff", "#d8cfff", "#b9a8ff", "#b7ddff", "#6da4d1", "#526f98"],
    },
  },
  {
    id: "black-opal",
    name: "Black Opal",
    hint: "Petrol depth · iridescent teal and violet",
    vars: {
      "--background": "#061012",
      "--surface": "#0d1b1e",
      "--foreground": "#e6f7f1",
      "--muted-foreground": "#7fa5a5",
      "--border": "#1d3b3d",
      "--primary": "#9be7cf",
      "--primary-foreground": "#061012",
      "--accent": "#b8a4ff",
    },
    palette: {
      paper: "#061012",
      ink: "#e6f7f1",
      edge: "rgba(91,211,193,0.22)",
      edgeHot: "rgba(231,255,247,0.96)",
      layers: {
        evidence: "#e6f7f1",
        games: "#9be7cf",
        event: "#b8a4ff",
        sport: "#58bfc0",
        venue: "#d8caff",
        answer: "#4e858d",
      },
      chroma: ["#e6f7f1", "#d8caff", "#b8a4ff", "#9be7cf", "#58bfc0", "#4e858d"],
    },
  },
  {
    id: "architectural-constellation",
    name: "Architectural Constellation",
    hint: "Midnight slate · blueprint cyan · measured amber",
    vars: {
      "--background": "#0b1119",
      "--surface": "#121c28",
      "--foreground": "#e8f0f5",
      "--muted-foreground": "#8093a3",
      "--border": "#263b4c",
      "--primary": "#a9d9e8",
      "--primary-foreground": "#0b1119",
      "--accent": "#e0b66e",
    },
    palette: {
      paper: "#0b1119",
      ink: "#e8f0f5",
      edge: "rgba(142,203,222,0.2)",
      edgeHot: "rgba(255,226,164,0.94)",
      layers: { evidence: "#e8f0f5", games: "#e0b66e", event: "#8ecbde", sport: "#7897b3", venue: "#f0cf91", answer: "#526d82" },
      chroma: ["#e8f0f5", "#f0cf91", "#e0b66e", "#8ecbde", "#7897b3", "#526d82"],
    },
  },
  {
    id: "nocturne",
    name: "Nocturne",
    hint: "Indigo plum · cold silver signal",
    vars: {
      "--background": "#0a0812",
      "--surface": "#120f1d",
      "--foreground": "#e9e4f2",
      "--muted-foreground": "#8b829e",
      "--border": "#221c31",
      "--primary": "#e9e4f2",
      "--primary-foreground": "#0a0812",
      "--accent": "#9d7bd8",
    },
    palette: {
      paper: "#0a0812",
      ink: "#e9e4f2",
      edge: "rgba(190,175,225,0.12)",
      edgeHot: "rgba(226,214,255,0.76)",
      layers: {
        evidence: "#efeaf8",
        games: "#9d7bd8",
        event: "#6f8ad8",
        sport: "#c48bb4",
        venue: "#d8cf9a",
        answer: "#5b5470",
      },
      chroma: ["#efeaf8", "#9d7bd8", "#6f8ad8", "#c48bb4", "#7fc0c4", "#d8cf9a"],
    },
  },
  {
    id: "cobalt",
    name: "Cobalt",
    hint: "Deep sea navy · ice blue signal",
    vars: {
      "--background": "#03060d",
      "--surface": "#0a1120",
      "--foreground": "#dfeaff",
      "--muted-foreground": "#7f92b5",
      "--border": "#15203a",
      "--primary": "#6fa8ff",
      "--primary-foreground": "#03060d",
      "--accent": "#3c76d8",
    },
    palette: {
      paper: "#03060d",
      ink: "#dfeaff",
      edge: "rgba(111,168,255,0.2)",
      edgeHot: "rgba(212,232,255,0.92)",
      layers: {
        evidence: "#eaf3ff",
        games: "#6fa8ff",
        event: "#3c76d8",
        sport: "#2a4f9e",
        venue: "#a9cbff",
        answer: "#17305e",
      },
      chroma: ["#eaf3ff", "#a9cbff", "#6fa8ff", "#3c76d8", "#2a4f9e", "#17305e"],
    },
  },
  {
    id: "eclipse",
    name: "Eclipse",
    hint: "Night ground · ember corona",
    vars: {
      "--background": "#08070a",
      "--surface": "#100e13",
      "--foreground": "#f4e6dc",
      "--muted-foreground": "#998a83",
      "--border": "#241f27",
      "--primary": "#ff9153",
      "--primary-foreground": "#08070a",
      "--accent": "#e2663a",
    },
    palette: {
      paper: "#08070a",
      ink: "#f4e6dc",
      edge: "rgba(255,145,83,0.12)",
      edgeHot: "rgba(255,178,120,0.78)",
      layers: {
        evidence: "#ffd7b8",
        games: "#ff9153",
        event: "#e2663a",
        sport: "#a24730",
        venue: "#fff2e4",
        answer: "#6a3324",
      },
      chroma: ["#ffd7b8", "#ffab68", "#ff8a45", "#e2663a", "#b04a2c", "#7a3320"],
    },
  },
  {
    id: "monochrome-grain",
    name: "Monochrome Grain",
    hint: "Graphite field · silver grain",
    vars: {
      "--background": "#090a0b",
      "--surface": "#111315",
      "--foreground": "#e8e9e7",
      "--muted-foreground": "#85898a",
      "--border": "#292d2e",
      "--primary": "#e8e9e7",
      "--primary-foreground": "#090a0b",
      "--accent": "#b9bdba",
    },
    palette: {
      paper: "#090a0b",
      ink: "#e8e9e7",
      edge: "rgba(232,233,231,0.14)",
      edgeHot: "rgba(255,255,255,0.86)",
      layers: {
        evidence: "#e8e9e7",
        games: "#c8ccca",
        event: "#aeb3b1",
        sport: "#949a99",
        venue: "#f4f5f3",
        answer: "#707676",
      },
      chroma: ["#f4f5f3", "#e8e9e7", "#c8ccca", "#aeb3b1", "#949a99", "#707676"],
    },
  },
  {
    id: "abyssal-opal",
    name: "Abyssal Opal",
    hint: "Petrol depth · pearl, cyan and violet signal",
    vars: {
      "--background": "#07191c",
      "--surface": "#0d282b",
      "--foreground": "#e8f6f2",
      "--muted-foreground": "#91b8b8",
      "--border": "#204447",
      "--primary": "#8ee7d2",
      "--primary-foreground": "#07191c",
      "--accent": "#bba7ff",
    },
    palette: {
      paper: "#07191c",
      ink: "#e8f6f2",
      edge: "rgba(115,207,198,0.34)",
      edgeHot: "rgba(231,255,247,0.96)",
      layers: {
        evidence: "#e8f6f2",
        games: "#8ee7d2",
        event: "#bba7ff",
        sport: "#66b9c0",
        venue: "#d3c4ff",
        answer: "#4d7883",
      },
      chroma: ["#e8f6f2", "#d3c4ff", "#bba7ff", "#8ee7d2", "#66b9c0", "#4d7883"],
    },
  },
  {
    id: "gilt",
    name: "Gilt",
    hint: "Lacquer black · leaf gold",
    vars: {
      "--background": "#070604",
      "--surface": "#120f09",
      "--foreground": "#f5ead2",
      "--muted-foreground": "#a29170",
      "--border": "#241d12",
      "--primary": "#e6bf6a",
      "--primary-foreground": "#070604",
      "--accent": "#b8873a",
    },
    palette: {
      paper: "#070604",
      ink: "#f5ead2",
      edge: "rgba(230,191,106,0.18)",
      edgeHot: "rgba(255,238,199,0.92)",
      layers: {
        evidence: "#faf1dd",
        games: "#e6bf6a",
        event: "#b8873a",
        sport: "#8a6224",
        venue: "#f3d9a1",
        answer: "#4a3517",
      },
      chroma: ["#faf1dd", "#f3d9a1", "#e6bf6a", "#b8873a", "#8a6224", "#4a3517"],
    },
  },
  {
    id: "bone",
    name: "Bone",
    hint: "Paper white · press black",
    vars: {
      "--background": "#f3f1ea",
      "--surface": "#eae7dd",
      "--foreground": "#16150f",
      "--muted-foreground": "#6d6a5c",
      "--border": "#cbc6b6",
      "--primary": "#16150f",
      "--primary-foreground": "#f3f1ea",
      "--accent": "#7a4b25",
    },
    palette: {
      paper: "#f3f1ea",
      ink: "#16150f",
      edge: "rgba(22,21,15,0.16)",
      edgeHot: "rgba(22,21,15,0.8)",
      layers: {
        evidence: "#16150f",
        games: "#7a4b25",
        event: "#9a7b3f",
        sport: "#4a5a49",
        venue: "#000000",
        answer: "#8a8574",
      },
      chroma: ["#16150f", "#7a4b25", "#9a7b3f", "#4a5a49", "#3c4c63", "#8a8574"],
    },
  },
  {
    id: "ivory-ledger",
    name: "Ivory Ledger",
    hint: "Archival ivory · ink black · copper accounting marks",
    vars: {
      "--background": "#f3efe5",
      "--surface": "#ebe4d6",
      "--foreground": "#171513",
      "--muted-foreground": "#756e62",
      "--border": "#c9beaa",
      "--primary": "#171513",
      "--primary-foreground": "#f3efe5",
      "--accent": "#9a4d32",
    },
    palette: {
      paper: "#f3efe5",
      ink: "#171513",
      edge: "rgba(23,21,19,0.22)",
      edgeHot: "rgba(23,21,19,0.82)",
      layers: {
        evidence: "#171513",
        games: "#9a4d32",
        event: "#7c2637",
        sport: "#6f7454",
        venue: "#a66b2d",
        answer: "#4f555b",
      },
      chroma: ["#171513", "#a66b2d", "#9a4d32", "#7c2637", "#6f7454", "#4f555b"],
    },
  },
  {
    id: "museum-archive",
    name: "Museum Archive",
    hint: "Warm vellum · ink black · oxblood provenance",
    vars: {
      "--background": "#eee8dc",
      "--surface": "#e4dbcc",
      "--foreground": "#201d1a",
      "--muted-foreground": "#776f63",
      "--border": "#c2b5a2",
      "--primary": "#201d1a",
      "--primary-foreground": "#eee8dc",
      "--accent": "#7d3040",
    },
    palette: {
      paper: "#eee8dc",
      ink: "#201d1a",
      edge: "rgba(32,29,26,0.2)",
      edgeHot: "rgba(125,48,64,0.88)",
      layers: { evidence: "#201d1a", games: "#a76a32", event: "#7d3040", sport: "#64705a", venue: "#9a5928", answer: "#55514b" },
      chroma: ["#201d1a", "#a76a32", "#7d3040", "#64705a", "#9a5928", "#55514b"],
    },
  },
  {
    id: "paper",
    name: "Paper White",
    hint: "Pure sheet · full spectrum plot",
    vars: {
      "--background": "#ffffff",
      "--surface": "#f5f5f6",
      "--foreground": "#101113",
      "--muted-foreground": "#74777d",
      "--border": "#e1e2e5",
      "--primary": "#101113",
      "--primary-foreground": "#ffffff",
      "--accent": "#2f6f6a",
    },
    palette: {
      paper: "#ffffff",
      ink: "#101113",
      edge: "rgba(16,17,19,0.13)",
      edgeHot: "rgba(16,17,19,0.68)",
      layers: {
        evidence: "#101113",
        games: "#c2483f",
        event: "#d68a2a",
        sport: "#2f6f6a",
        venue: "#1f3f7a",
        answer: "#8e8f95",
      },
      chroma: ["#c2483f", "#d68a2a", "#b8a52d", "#2f6f6a", "#1f3f7a", "#7a3f75"],
    },
  },
];

/* ------------------------------------------------------------------ *
 * Styles: how the projection is drawn
 * ------------------------------------------------------------------ */

export type NodeMark =
  | "soma"
  | "orb"
  | "vertex"
  | "moon"
  | "burst"
  | "planet"
  | "ganglion"
  | "astro"
  | "arbor";
export type EdgeMark =
  | "dendrite"
  | "spectral"
  | "mesh"
  | "hair"
  | "filament"
  | "axon"
  | "varicose"
  | "tendril";
/** global formation the layout is pulled into */
export type Formation = "free" | "brain" | "moon" | "orbit" | "culture";


export interface VizProfile {
  id: string;
  name: string;
  hint: string;
  nodeMark: NodeMark;
  edgeMark: EdgeMark;
  /** layer = 6 semantic hues · chroma = per-node spectral index */
  colorMode: "layer" | "chroma";
  labels: "hover" | "hubs" | "all";
  particles: boolean;
  charge: number;
  linkDistance: number;
  scale: number;
  formation: Formation;
  /** how hard nodes are pulled onto the formation (0 = free) */
  formStrength: number;
  /** theme auto-selected when this style is picked (user can still override) */
  preferredTheme: string;
}

export const VIZ_PROFILES: VizProfile[] = [
  {
    id: "orrery",
    name: "Orrery",
    hint: "Planets, rings and stars on orbits",
    nodeMark: "planet",
    edgeMark: "hair",
    colorMode: "chroma",
    labels: "hubs",
    particles: true,
    charge: -120,
    linkDistance: 90,
    scale: 1.45,
    formation: "orbit",
    formStrength: 0.72,
    preferredTheme: "obsidian-observatory",
  },
  {
    id: "obsidian-observatory",
    name: "Obsidian Observatory",
    hint: "Platinum bodies · champagne orbital traces",
    nodeMark: "planet",
    edgeMark: "hair",
    colorMode: "chroma",
    labels: "hubs",
    particles: true,
    charge: -120,
    linkDistance: 92,
    scale: 1.25,
    formation: "orbit",
    formStrength: 0.78,
    preferredTheme: "obsidian-observatory",
  },
  {
    id: "selene",
    name: "Selene",
    hint: "Crescent formation · phase-lit bodies",
    nodeMark: "moon",
    edgeMark: "hair",
    colorMode: "layer",
    labels: "hover",
    particles: true,
    charge: -70,
    linkDistance: 58,
    scale: 1.05,
    formation: "moon",
    formStrength: 0.72,
    preferredTheme: "nocturne",
  },
  {
    id: "nova",
    name: "Nova",
    hint: "Radial spokes · tapered filaments",
    nodeMark: "burst",
    edgeMark: "filament",
    colorMode: "chroma",
    labels: "hubs",
    particles: true,
    charge: -520,
    linkDistance: 168,
    scale: 1.12,
    formation: "free",
    formStrength: 0,
    preferredTheme: "gilt",
  },
  {
    id: "lunar",
    name: "Lunar Phase",
    hint: "Phase-lit moons · hairline sky",
    nodeMark: "moon",
    edgeMark: "hair",
    colorMode: "layer",
    labels: "hover",
    particles: true,
    charge: -430,
    linkDistance: 190,
    scale: 1.3,
    formation: "free",
    formStrength: 0,
    preferredTheme: "eclipse",
  },
  {
    id: "chromatic",
    name: "Chromatic Web",
    hint: "Hue-indexed orbs · blended strands",
    nodeMark: "orb",
    edgeMark: "spectral",
    colorMode: "chroma",
    labels: "hubs",
    particles: false,
    charge: -330,
    linkDistance: 118,
    scale: 1.25,
    formation: "free",
    formStrength: 0,
    preferredTheme: "obsidian-observatory",
  },
  {
    id: "glass-observatory",
    name: "Glass Observatory",
    hint: "Parallax moons · luminous hairlines · sparse depth",
    nodeMark: "moon",
    edgeMark: "hair",
    colorMode: "chroma",
    labels: "hubs",
    particles: true,
    charge: -460,
    linkDistance: 178,
    scale: 1.24,
    formation: "orbit",
    formStrength: 0.58,
    preferredTheme: "glass-observatory",
  },
  {
    id: "architectural-constellation",
    name: "Architectural Constellation",
    hint: "Blueprint rings · precise vertices · measured links",
    nodeMark: "vertex",
    edgeMark: "mesh",
    colorMode: "layer",
    labels: "hubs",
    particles: true,
    charge: -300,
    linkDistance: 122,
    scale: 1.08,
    formation: "orbit",
    formStrength: 0.8,
    preferredTheme: "architectural-constellation",
  },
  {
    id: "black-opal",
    name: "Black Opal",
    hint: "Iridescent orbs · restrained spectral shimmer",
    nodeMark: "orb",
    edgeMark: "spectral",
    colorMode: "chroma",
    labels: "hubs",
    particles: true,
    charge: -240,
    linkDistance: 104,
    scale: 1.18,
    formation: "culture",
    formStrength: 0.65,
    preferredTheme: "black-opal",
  },
  {
    id: "fabric",
    name: "Space Fabric",
    hint: "Wireframe vertices · taut mesh",
    nodeMark: "vertex",
    edgeMark: "mesh",
    colorMode: "layer",
    labels: "hubs",
    particles: false,
    charge: -210,
    linkDistance: 86,
    scale: 0.92,
    formation: "free",
    formStrength: 0,
    preferredTheme: "cobalt",
  },
  {
    id: "neuron",
    name: "Neuron",
    hint: "Somas · dendrite arbors",
    nodeMark: "soma",
    edgeMark: "dendrite",
    colorMode: "layer",
    labels: "hover",
    particles: false,
    charge: -340,
    linkDistance: 128,
    scale: 1.1,
    formation: "free",
    formStrength: 0,
    preferredTheme: "monochrome-grain",
  },
  {
    id: "museum-archive",
    name: "Museum Archive",
    hint: "Archival grid · provenance marks · quiet evidence density",
    nodeMark: "vertex",
    edgeMark: "axon",
    colorMode: "layer",
    labels: "all",
    particles: false,
    charge: -240,
    linkDistance: 102,
    scale: 0.92,
    formation: "culture",
    formStrength: 0.48,
    preferredTheme: "museum-archive",
  },
];

/* ------------------------------------------------------------------ *
 * Formations — deterministic target points sampled from a silhouette
 * ------------------------------------------------------------------ */

const R = 340;

/** van der Corput radical inverse — even, deterministic point spread */
function halton(i: number, base: number) {
  let f = 1;
  let r = 0;
  let k = i + 1;
  while (k > 0) {
    f /= base;
    r += f * (k % base);
    k = Math.floor(k / base);
  }
  return r;
}

function seeded(id: string, salt: number) {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < id.length; i++) h = (h ^ id.charCodeAt(i)) * 16777619;
  h >>>= 0;
  return () => {
    h = (h * 1664525 + 1013904223) >>> 0;
    return h / 4294967296;
  };
}

/* ---------------- brain silhouette (sagittal, facing left) ----------------
 * A hand-drawn anatomical contour: frontal pole, parietal crown, occipital
 * curve, preoccipital notch, cerebellum, brain stem and temporal lobe.
 * Nodes ride the contour and a concentric inner ribbon, so the silhouette
 * reads as cortex — outline plus cortical band, never a filled blob.        */

const BRAIN: [number, number][] = [
  [-1.0, 0.02],
  [-0.97, -0.28],
  [-0.83, -0.5],
  [-0.56, -0.66],
  [-0.2, -0.74],
  [0.2, -0.68],
  [0.54, -0.5],
  [0.79, -0.22],
  [0.86, 0.06],
  [0.72, 0.19],
  [0.87, 0.33],
  [0.8, 0.55],
  [0.6, 0.66],
  [0.41, 0.57],
  [0.31, 0.73],
  [0.17, 0.88],
  [0.03, 0.84],
  [0.11, 0.6],
  [0.06, 0.43],
  [-0.16, 0.52],
  [-0.47, 0.5],
  [-0.72, 0.35],
  [-0.93, 0.16],
];

const BRAIN_CX = 0.0;
const BRAIN_CY = -0.02;

const BRAIN_SEG = (() => {
  const seg: number[] = [];
  let total = 0;
  for (let i = 0; i < BRAIN.length; i++) {
    const a = BRAIN[i]!;
    const b = BRAIN[(i + 1) % BRAIN.length]!;
    const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
    seg.push(d);
    total += d;
  }
  return { seg, total };
})();

/** evenly spaced point along the contour, optionally shrunk toward the centroid */
function brainContour(k: number, total: number, shrink = 1) {
  const u = (k % Math.max(1, total)) / Math.max(1, total);
  let d = u * BRAIN_SEG.total;
  let i = 0;
  while (i < BRAIN_SEG.seg.length - 1 && d > BRAIN_SEG.seg[i]!) {
    d -= BRAIN_SEG.seg[i]!;
    i++;
  }
  const a = BRAIN[i]!;
  const b = BRAIN[(i + 1) % BRAIN.length]!;
  const t = d / Math.max(1e-6, BRAIN_SEG.seg[i]!);
  const px = a[0] + (b[0] - a[0]) * t;
  const py = a[1] + (b[1] - a[1]) * t;
  return {
    x: BRAIN_CX + (px - BRAIN_CX) * shrink,
    y: BRAIN_CY + (py - BRAIN_CY) * shrink,
  };
}

export function formationTargets(
  formation: Formation,
  nodes: { id: string }[],
): Map<string, { x: number; y: number }> {
  const out = new Map<string, { x: number; y: number }>();
  const n = Math.max(1, nodes.length);
  nodes.forEach((node, i) => {
    const r = seeded(node.id, formation.length * 7 + 11);
    let x = 0;
    let y = 0;
    if (formation === "brain") {
      /* three concentric bands: silhouette, cortical ribbon, deep tissue */
      const band = i % 5;
      const shrink = band < 3 ? 1 : band === 3 ? 0.72 : 0.45;
      const per = band < 3 ? Math.ceil((n * 3) / 5) : Math.ceil(n / 5);
      const k = band < 3 ? Math.floor(i / 5) * 3 + band : Math.floor(i / 5);
      const off = band === 3 ? 0.5 : band === 4 ? 1.4 : 0;
      const p = brainContour(k + off, Math.max(8, per), shrink);
      x = p.x * 1.4;
      y = p.y * 1.4;
    } else if (formation === "moon") {

      if (i % 3 === 0) {
        // outer limb
        const t = (Math.floor(i / 3) / Math.ceil(n / 3)) * Math.PI * 2;
        x = Math.cos(t) * 1.25;
        y = Math.sin(t) * 1.25;
      } else {
        for (let t = 0; t < 96; t++) {
          const px = halton(i * 47 + t, 2) * 2 - 1;
          const py = halton(i * 47 + t, 3) * 2 - 1;
          const inOuter = px * px + py * py <= 1;
          const inCut = (px - 0.42) ** 2 + (py + 0.06) ** 2 <= 0.72 ** 2;
          if (inOuter && !inCut) {
            x = px * 1.25;
            y = py * 1.25;
            break;
          }
        }
      }
    } else if (formation === "orbit") {
      const rings = 5;
      const ring = i % rings;
      const per = Math.ceil(n / rings);
      const a = ((Math.floor(i / rings) % per) / per) * Math.PI * 2 + ring * 0.7;
      const rad = 0.32 + ring * 0.2 + (r() - 0.5) * 0.045;
      x = Math.cos(a) * rad * 1.5;
      y = Math.sin(a) * rad * 0.92;
    } else if (formation === "culture") {
      /* cultured tissue field: jittered hex lattice, evenly seeded */
      const cols = Math.max(3, Math.round(Math.sqrt(n * 2.1)));
      const rows = Math.ceil(n / cols);
      const cx = i % cols;
      const cy = Math.floor(i / cols);
      const stagger = cy % 2 ? 0.5 : 0;
      x = ((cx + 0.5 + stagger) / cols - 0.5) * 2.9 + (r() - 0.5) * 0.16;
      y = ((cy + 0.5) / Math.max(1, rows) - 0.5) * 1.85 + (r() - 0.5) * 0.16;
    } else {
      return;
    }
    out.set(node.id, { x: x * R, y: y * R });
  });
  return out;
}
