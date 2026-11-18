import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Html, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { Edge, Point, Relation } from "../api";

const SCALE = 1.7;
const COLORS: Record<Point["kind"], string> = { query: "#f2bf4b", neighbor: "#38c48f", token: "#b9d9cb", sentence: "#ff3d5a" };

interface Props {
  points: Point[];
  edges: Edge[];
  selected: string | null;
  autoRotate: boolean;
  onSelect: (p: Point) => void;
  axisNames: [string, string, string];
  dims: 2 | 3;
  pair: [string | null, string | null];
  pairCos: number | null;
  relations: Relation[];
}

const AXIS = [
  { dir: new THREE.Vector3(1, 0, 0), color: "#ff7a7a", name: "X" },
  { dir: new THREE.Vector3(0, 1, 0), color: "#7be0a0", name: "Y" },
  { dir: new THREE.Vector3(0, 0, 1), color: "#7fb2ff", name: "Z" },
];
const AXIS_LEN = SCALE * 1.15;

/** X, Y, Z axes of the projection: line, arrow head, ticks at -1, -0.5, 0.5, 1 (projection units) and a name label. */
function Axes({ names, dims }: { names: [string, string, string]; dims: 2 | 3 }) {
  const shown = AXIS.slice(0, dims);
  return (
    <group>
      {shown.map((a, i) => {
        const pts = new Float32Array([...a.dir.clone().multiplyScalar(-AXIS_LEN).toArray(), ...a.dir.clone().multiplyScalar(AXIS_LEN).toArray()]);
        const tip = a.dir.clone().multiplyScalar(AXIS_LEN);
        const rot: [number, number, number] = i === 0 ? [0, 0, -Math.PI / 2] : i === 2 ? [Math.PI / 2, 0, 0] : [0, 0, 0];
        return (
          <group key={a.name}>
            <line>
              <bufferGeometry><bufferAttribute attach="attributes-position" args={[pts, 3]} /></bufferGeometry>
              <lineBasicMaterial color={a.color} transparent opacity={0.55} />
            </line>
            <mesh position={tip} rotation={rot}><coneGeometry args={[0.03, 0.09, 12]} /><meshBasicMaterial color={a.color} /></mesh>
            {[-1, -0.5, 0.5, 1].map((t) => (
              <group key={t} position={a.dir.clone().multiplyScalar(t * SCALE)}>
                <mesh><sphereGeometry args={[0.009, 8, 8]} /><meshBasicMaterial color={a.color} /></mesh>
                <Html center zIndexRange={[5, 0]} style={{ pointerEvents: "none" }}>
                  <div className="axis-tick" style={{ color: a.color }}>{t}</div>
                </Html>
              </group>
            ))}
            <Html position={tip.clone().multiplyScalar(1.09)} center zIndexRange={[6, 0]} style={{ pointerEvents: "none" }}>
              <div className="axis-label" style={{ borderColor: a.color, color: a.color }}><b>{a.name}</b> {names[i]}</div>
            </Html>
          </group>
        );
      })}
      <Html position={[0, 0, 0]} center zIndexRange={[5, 0]} style={{ pointerEvents: "none" }}><div className="axis-tick">0</div></Html>
    </group>
  );
}

export default function Scene(props: Props) {
  return (
    <Canvas camera={{ position: [0, 0.35, 4.1], fov: 52 }} dpr={[1, 2]} gl={{ antialias: true }}>
      <color attach="background" args={["#07110e"]} />
      <fog attach="fog" args={["#07110e", 3.4, 8.5]} />
      <ambientLight intensity={0.6} />
      <pointLight position={[3, 4, 5]} intensity={40} color="#ffe2a8" />
      <Axes names={props.axisNames} dims={props.dims} />
      <Graph {...props} />
      <OrbitControls enableDamping dampingFactor={0.08} autoRotate={props.autoRotate} autoRotateSpeed={0.35} minDistance={1.2} maxDistance={8} />
    </Canvas>
  );
}

function Graph({ points, edges, selected, onSelect, pair, pairCos, relations }: Props) {
  // current (animated) positions, shared by nodes and edges
  const cur = useRef(new Map<string, THREE.Vector3>());
  const target = useMemo(() => new Map(points.map((p) => [p.id, new THREE.Vector3(...p.xyz).multiplyScalar(SCALE)])), [points]);
  useEffect(() => {
    for (const [id, v] of target) if (!cur.current.has(id)) cur.current.set(id, v.clone().multiplyScalar(0.05));
    for (const id of [...cur.current.keys()]) if (!target.has(id)) cur.current.delete(id);
  }, [target]);

  const lineGeo = useMemo(() => new THREE.BufferGeometry(), []);
  const pairGeo = useMemo(() => { const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3)); return g; }, []);
  const pairLine = useRef<THREE.Line>(null);
  const pairLabel = useRef<THREE.Group>(null);
  const pairReady = !!(pair[0] && pair[1] && target.has(pair[0]) && target.has(pair[1]));
  const lineRef = useRef<THREE.LineSegments>(null);
  // relationship lines between the user's own words / sentences (cosine from the full vectors)
  const relGeo = useMemo(() => new THREE.BufferGeometry(), []);
  const relLabels = useRef<(THREE.Group | null)[]>([]);
  const relShown = relations.filter((r) => target.has(r.source) && target.has(r.target));
  useEffect(() => {
    const col = new Float32Array(relShown.length * 6);
    relShown.forEach((r, i) => {
      const k = 0.2 + 0.8 * Math.max(0, Math.min(1, r.cos / 0.7));
      const c = new THREE.Color("#8fd3ff").multiplyScalar(k);
      col.set([c.r, c.g, c.b, c.r, c.g, c.b], i * 6);
    });
    relGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(relShown.length * 6), 3));
    relGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  }, [relShown.map((r) => r.source + r.target + r.cos).join(","), relGeo]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const arr = new Float32Array(edges.length * 6); const col = new Float32Array(edges.length * 6);
    edges.forEach((e, i) => {
      const s = e.source.startsWith("s:") ? new THREE.Color("#ff3d5a") : new THREE.Color("#f2bf4b");
      const k = 0.25 + 0.75 * Math.max(0, e.cos);
      col.set([s.r * k, s.g * k, s.b * k, s.r * k * 0.4, s.g * k * 0.4, s.b * k * 0.4], i * 6);
    });
    lineGeo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    lineGeo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  }, [edges, lineGeo]);

  useFrame((_, dt) => {
    const a = 1 - Math.pow(0.0015, dt);              // frame-rate independent easing
    for (const [id, v] of cur.current) { const t = target.get(id); if (t) v.lerp(t, a); }
    const pos = lineGeo.getAttribute("position") as THREE.BufferAttribute | undefined;
    if (!pos) return;
    edges.forEach((e, i) => {
      const s = cur.current.get(e.source), t = cur.current.get(e.target);
      if (s && t) { pos.setXYZ(i * 2, s.x, s.y, s.z); pos.setXYZ(i * 2 + 1, t.x, t.y, t.z); }
    });
    pos.needsUpdate = true;
    const rp = relGeo.getAttribute("position") as THREE.BufferAttribute | undefined;
    if (rp && rp.count === relShown.length * 2) {
      relShown.forEach((r, i) => {
        const s = cur.current.get(r.source), t = cur.current.get(r.target);
        if (!s || !t) return;
        rp.setXYZ(i * 2, s.x, s.y, s.z); rp.setXYZ(i * 2 + 1, t.x, t.y, t.z);
        relLabels.current[i]?.position.copy(s).add(t).multiplyScalar(0.5);
      });
      rp.needsUpdate = true;
    }
    // similarity line between the two picked points
    const pa = pair[0] ? cur.current.get(pair[0]) : undefined, pb = pair[1] ? cur.current.get(pair[1]) : undefined;
    const pp = pairGeo.getAttribute("position") as THREE.BufferAttribute;
    if (pa && pb) {
      pp.setXYZ(0, pa.x, pa.y, pa.z); pp.setXYZ(1, pb.x, pb.y, pb.z); pp.needsUpdate = true;
      pairLine.current?.computeLineDistances();
      if (pairLabel.current) pairLabel.current.position.copy(pa).add(pb).multiplyScalar(0.5);
    }
  });

  const selGroups = useMemo(() => { const p = points.find((x) => x.id === selected); return p ? new Set(p.groups) : null; }, [points, selected]);
  return (
    <group>
      <lineSegments ref={lineRef} geometry={lineGeo}>
        <lineBasicMaterial vertexColors transparent opacity={0.75} depthWrite={false} />
      </lineSegments>
      <lineSegments geometry={relGeo}>
        <lineBasicMaterial vertexColors transparent opacity={0.9} depthWrite={false} />
      </lineSegments>
      {relShown.map((r, i) => (
        <group key={r.source + "|" + r.target} ref={(g) => { relLabels.current[i] = g; }}>
          <Html center zIndexRange={[18, 0]} style={{ pointerEvents: "none" }}>
            <div className="rel-label" style={{ opacity: relShown.length > 15 && !(selected === r.source || selected === r.target) ? 0 : 1 }}>{r.cos.toFixed(2)}</div>
          </Html>
        </group>
      ))}
      {pairReady && (
        <>
          {/* @ts-expect-error three line element */}
          <line ref={pairLine} geometry={pairGeo}>
            <lineDashedMaterial color="#ffffff" dashSize={0.06} gapSize={0.04} transparent opacity={0.9} />
          </line>
          <group ref={pairLabel}>
            <Html center zIndexRange={[25, 0]} style={{ pointerEvents: "none" }}>
              <div className="pair-label">cos {pairCos === null ? "–" : pairCos.toFixed(3)}</div>
            </Html>
          </group>
        </>
      )}
      {points.map((p) => (
        <Node key={p.id} p={p} pos={cur} selected={selected === p.id || pair.includes(p.id)} onSelect={onSelect}
          related={!!selGroups && p.groups.some((g) => selGroups.has(g))} />
      ))}
    </group>
  );
}

function Node({ p, pos, selected, related, onSelect }: { p: Point; pos: React.MutableRefObject<Map<string, THREE.Vector3>>; selected: boolean; related: boolean; onSelect: (p: Point) => void }) {
  const ref = useRef<THREE.Group>(null);
  const born = useRef(performance.now());
  const [hov, setHov] = useState(false);
  const size = p.kind === "query" ? 0.055 : p.kind === "sentence" ? 0.07 : p.kind === "neighbor" ? 0.032 : 0.026;
  useFrame(() => {
    const g = ref.current; const v = pos.current.get(p.id);
    if (!g || !v) return;
    g.position.copy(v);
    const age = Math.min(1, (performance.now() - born.current) / 700);
    const pop = 1 - Math.pow(1 - age, 3);
    const pulse = selected ? 1.35 + 0.12 * Math.sin(performance.now() / 220) : hov ? 1.3 : 1;
    g.scale.setScalar(pop * pulse);
  });
  const showLabel = p.kind === "query" || p.kind === "sentence" || hov || selected || related;
  return (
    <group ref={ref}>
      <mesh
        onClick={(e) => { e.stopPropagation(); onSelect(p); }}
        onPointerOver={(e) => { e.stopPropagation(); setHov(true); document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { setHov(false); document.body.style.cursor = ""; }}>
        {p.kind === "sentence" ? <octahedronGeometry args={[size, 0]} /> : <sphereGeometry args={[size, 20, 20]} />}
        <meshStandardMaterial color={COLORS[p.kind]} emissive={COLORS[p.kind]} emissiveIntensity={p.kind === "query" || selected ? 0.9 : 0.45} roughness={0.35} />
      </mesh>
      {(p.kind === "query" || p.kind === "sentence") && (
        <mesh scale={2.6}><sphereGeometry args={[size, 16, 16]} /><meshBasicMaterial color={COLORS[p.kind]} transparent opacity={0.12} depthWrite={false} /></mesh>
      )}
      {showLabel && (
        <Html center={false} zIndexRange={[15, 0]} style={{ pointerEvents: "none" }}>
          <div className={`label3d ${p.kind === "query" ? "query" : p.kind === "sentence" ? "sentence" : ""}`} style={{ opacity: p.kind === "neighbor" && !hov && !selected ? 0.8 : 1 }}>
            {p.label}
          </div>
        </Html>
      )}
    </group>
  );
}
