import { useEffect, useRef, useState } from "react";
import * as $3Dmol from "3dmol";
import { HIT_CARBONS, QUERY_CARBON, QUERY_SURFACE, elementHex, overlayStyle } from "./viewer-style";
export type Overlay = {
  hit: string;
  query: string;
  row: { rank: number; query_index: number; name: string };
};
export function Viewer({ items }: { items: Overlay[] }) {
  const element = useRef<HTMLDivElement>(null);
  const viewer = useRef<$3Dmol.GLViewer | null>(null);
  const [style, setStyle] = useState("surface");
  const [coloring, setColoring] = useState("element");
  const [query, setQuery] = useState(true);
  const [hits, setHits] = useState(true);
  const [surfaceState, setSurfaceState] = useState("none");
  const [surfaceError, setSurfaceError] = useState("");
  useEffect(() => {
    if (!element.current) return;
    const v = $3Dmol.createViewer(element.current, {
      backgroundColor: "#f6f8fb",
    });
    viewer.current = v;
    const observer = new ResizeObserver(() => {
      v.resize();
      v.render();
    });
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      v.clear();
      viewer.current = null;
    };
  }, []);
  useEffect(() => {
    const v = viewer.current;
    if (!v) return;
    let stopped = false;
    v.clear();
    setSurfaceError("");
    const seenQueries = new Set<string>();
    const queryModels: $3Dmol.GLModel[] = [];
    items.forEach((item, i) => {
      // Pinned hits can share a query conformer. Drawing its translucent
      // envelope repeatedly would change the apparent opacity.
      if (query && !seenQueries.has(item.query)) {
        seenQueries.add(item.query);
        const m = v.addModel(item.query, "sdf");
        queryModels.push(m);
        m.setStyle(
          {},
          overlayStyle(style, QUERY_CARBON, true, coloring === "element"),
        );
      }
      if (hits && item.row.rank !== 0) {
        const m = v.addModel(item.hit, "sdf");
        m.setStyle(
          {},
          overlayStyle(style, HIT_CARBONS[i % HIT_CARBONS.length], false, coloring === "element"),
        );
      }
    });
    if (style === "surface" && queryModels.length) {
      setSurfaceState("loading");
      Promise.all(queryModels.map((model) =>
        v.addSurface($3Dmol.SurfaceType.VDW, { ...QUERY_SURFACE }, { model }),
      )).then(() => {
        if (!stopped) {
          setSurfaceState("ready");
          v.render();
        }
      }).catch(() => {
        if (!stopped) {
          setSurfaceState("error");
          setSurfaceError("Query surface could not be drawn. Query sticks are still available.");
        }
      });
    } else setSurfaceState("none");
    v.zoomTo();
    v.render();
    return () => { stopped = true; };
  }, [items, style, coloring, query, hits]);
  return (
    <section className="viewer-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">ALIGNED POSES</span>
          <h2>Explore the overlap</h2>
        </div>
        <button
          className="quiet"
          onClick={() => {
            viewer.current?.zoomTo();
            viewer.current?.render();
          }}
        >
          Reset view
        </button>
      </div>
      <div ref={element} className="viewer" data-testid="molecular-viewer" data-surface-state={surfaceState} />
      <div className="viewer-tools">
        <label>
          <input
            type="checkbox"
            checked={query}
            onChange={(e) => setQuery(e.target.checked)}
          />{" "}
          Query
        </label>
        <label>
          <input
            type="checkbox"
            checked={hits}
            onChange={(e) => setHits(e.target.checked)}
          />{" "}
          Hits
        </label>
        <select
          aria-label="Coloring"
          value={coloring}
          onChange={(e) => setColoring(e.target.value)}
        >
          <option value="element">Element + carbon identity</option>
          <option value="molecule">Molecule colors</option>
        </select>
        <select
          aria-label="Representation"
          value={style}
          onChange={(e) => setStyle(e.target.value)}
        >
          <option value="stick">Sticks</option>
          <option value="sphere">Spheres</option>
          <option value="surface">Query surface + hit spheres</option>
        </select>
      </div>
      <div className="legend">
        {!!items.length && (
          <div>
            <span style={{ background: QUERY_CARBON }} />Query
            <small>{style === "surface" ? "gray surface + thin sticks" : coloring === "element" ? "carbon atoms" : "all atoms"}</small>
          </div>
        )}
        {items.map((x, i) => x.row.rank !== 0 && (
          <div key={x.row.rank}>
            <span style={{ background: HIT_CARBONS[i % HIT_CARBONS.length] }} />#{x.row.rank}{" "}
            {x.row.name || "Unnamed molecule"}{" "}
            <small>{coloring === "element" ? "carbon color; " : ""}query conformer {x.row.query_index + 1}</small>
          </div>
        ))}
      </div>
      {coloring === "element" && (
        <div className="element-legend" aria-label="Element colors">
          {(["N", "O", "S", "P", "F", "Cl", "Br"] as const).map((element) => (
            <div key={element}>
              <span style={{ background: elementHex(element) }} />{element}
            </div>
          ))}
        </div>
      )}
      {!items.length && (
        <p className="hint">
          Select a hit to load its optimized pose. Drag to rotate; scroll to
          zoom.
        </p>
      )}
      {items.length > 0 && items.every((item) => item.row.rank === 0) && (
        <p className="hint">Query preview · first prepared conformer. Select a hit to show its winning query conformer and optimized pose.</p>
      )}
      {surfaceError && <p role="alert">{surfaceError}</p>}
      {style === "surface" && (
        <p className="hint">The translucent query envelope is a visual van der Waals surface, not the Gaussian field used for scoring. Query sticks retain atom colors; hits are solid spheres.</p>
      )}
      <p className="hint">
        Each hit is paired with its winning query conformer in the prepared
        coordinate frame. {coloring === "element" && "Carbon colors identify molecules; other atoms use element colors."}
      </p>
    </section>
  );
}
