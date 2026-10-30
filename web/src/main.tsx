import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Viewer, type Overlay } from "./viewer";
import "./style.css";
type DB = {
  id: string;
  name: string;
  description: string;
  molecules: number;
  conformers: number;
  modes: string[];
  overlays: boolean;
};
type Job = {
  id: string;
  state: string;
  created: number;
  progress: number;
  submitted_conformers: number;
  completed_conformers?: number;
  progress_basis?: "completed" | "submitted";
  total_conformers?: number;
  eta_seconds?: number | null;
  error?: string;
  request: any;
  preparation?: { conformers: number; warnings: string[]; molblock: string };
};
type Hit = {
  rank: number;
  gid: string;
  mol_id: string;
  conf_idx: number;
  name: string;
  query_index: number;
  shape: number;
  color: number;
  combination: number;
};
function isActiveJob(job: Job | null): boolean {
  return !!job && ["draft", "preparing", "queued", "starting", "waiting_gpu", "waiting_cpu", "screening", "finalizing"].includes(job.state);
}
async function api(path: string, options?: RequestInit) {
  const response = await fetch("/api" + path, options);
  if (!response.ok) {
    const d = await response.json();
    throw new Error(
      typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail),
    );
  }
  return response.json();
}
function App() {
  const [dbs, setDbs] = useState<DB[]>([]),
    [database, setDatabase] = useState(""),
    [mode, setMode] = useState("shape");
  const [limits, setLimits] = useState({
    max_k: 10000,
    max_conformers: 50,
    retention_days: 7,
  });
  const [k, setK] = useState(1000),
    [mix, setMix] = useState(0.5),
    [smiles, setSmiles] = useState(""),
    [file, setFile] = useState<File | null>(null);
  const [generate, setGenerate] = useState(false),
    [count, setCount] = useState(10),
    [minimize, setMinimize] = useState(false);
  const [job, setJob] = useState<Job | null>(null),
    [jobs, setJobs] = useState<Job[]>([]),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [hits, setHits] = useState<Hit[]>([]),
    [total, setTotal] = useState(0),
    [page, setPage] = useState(0),
    [overlays, setOverlays] = useState<Overlay[]>([]),
    [pin, setPin] = useState(false);
  const currentJob = useRef<string | undefined>(job?.id);
  currentJob.current = job?.id;
  const overlayRequest = useRef(0);
  const previewItems = useMemo<Overlay[]>(() => job?.preparation ? [{
    query: job.preparation.molblock, hit: job.preparation.molblock,
    row: { rank: 0, name: "Query preview", query_index: 0 },
  }] : [], [job?.id, job?.preparation?.molblock]);
  const selected = dbs.find((x) => x.id === database);
  const hasActiveJobs = jobs.some(isActiveJob);
  async function attempt(fn: () => Promise<void>) {
    setError("");
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    api("/databases")
      .then((d) => {
        setDbs(d.databases);
        setDatabase(d.databases[0]?.id || "");
        setLimits(d.limits);
      })
      .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (selected && !selected.modes.includes(mode)) setMode("shape");
  }, [database]);
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    const refresh = async () => {
      try {
        const all: Job[] = await api("/jobs", { signal: controller.signal });
        if (stopped) return;
        setJobs(all);
        let next = job;
        if (isActiveJob(job)) {
          // History includes full job records; only fetch separately if this
          // selected job falls outside the bounded history window.
          next = all.find((item) => item.id === job!.id) ??
            await api("/jobs/" + job!.id, { signal: controller.signal });
          if (stopped) return;
          setJob(next);
        }
        if (all.some(isActiveJob) || isActiveJob(next))
          timer = setTimeout(refresh, 1500);
      } catch (e) {
        if (!stopped) {
          setError((e as Error).message);
          if (hasActiveJobs || isActiveJob(job)) timer = setTimeout(refresh, 1500);
        }
      }
    };
    refresh();
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller.abort();
    };
  }, [job?.id, job?.state, hasActiveJobs]);
  useEffect(() => {
    const ident = new URLSearchParams(location.search).get("job");
    if (ident)
      api("/jobs/" + ident)
        .then(setJob)
        .catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    if (job) history.replaceState(null, "", "?job=" + job.id);
  }, [job?.id]);
  useEffect(() => {
    setOverlays([]);
    setPage(0);
    setHits([]);
    setTotal(0);
    overlayRequest.current++;
  }, [job?.id]);
  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    if (job?.state === "completed")
      api(`/jobs/${job.id}/hits?offset=${page * 50}`, { signal: controller.signal })
        .then((d) => {
          if (stopped) return;
          setHits(d.rows);
          setTotal(d.total);
        })
        .catch((e) => { if (!stopped) setError(e.message); });
    return () => { stopped = true; controller.abort(); };
  }, [job?.id, job?.state, page]);
  const preparing = job && ["preparing", "draft"].includes(job.state);
  const running =
    job &&
    ["queued", "starting", "waiting_gpu", "waiting_cpu", "screening", "finalizing"].includes(
      job.state,
    );
  async function prepare() {
    const form = new FormData();
    if (file) form.append("file", file);
    else form.append("smiles", smiles);
    form.append("generate", String(generate));
    form.append("conformers", String(count));
    form.append("minimize", String(minimize));
    const next = await api("/queries", { method: "POST", body: form });
    setJob(next);
  }
  async function screen() {
    if (!job) return;
    setJob(
      await api("/jobs/" + job.id + "/screen", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ database, mode, k, mixing: mix }),
      }),
    );
  }
  async function choose(rank: number) {
    if (!job) return;
    const ident = job.id;
    const request = ++overlayRequest.current;
    let item;
    try {
      item = await api(`/jobs/${ident}/hits/${rank}/overlay`);
    } catch (error) {
      if (currentJob.current === ident && request === overlayRequest.current) throw error;
      return;
    }
    if (currentJob.current !== ident || request !== overlayRequest.current) return;
    setOverlays((old) =>
      pin
        ? [...old.filter((x) => x.row.rank !== rank), item].slice(-4)
        : [item],
    );
  }
  const elapsed = job ? Math.max(0, Date.now() / 1000 - job.created) : 0;
  const eta =
    job?.state === "screening" && typeof job.eta_seconds === "number"
      && Number.isFinite(job.eta_seconds) && job.eta_seconds >= 0
      ? Math.ceil(job.eta_seconds)
      : null;
  return (
    <div className="app">
      <header>
        <a href="/" className="brand">
          <span className="brand-mark">R₃</span>
          <span>
            ROSHAMBO<span className="brand-three">3</span>
          </span>
        </a>
        <span className="tag">MOLECULAR SHAPE SCREENING</span>
        <span className="status-dot" />
        Lab workspace
      </header>
      <main>
        <div className="intro">
          <span className="eyebrow">FROM STRUCTURE TO SIMILARITY</span>
          <h1>Find your molecule’s next match.</h1>
          <p>
            Search a molecular database, then explore the best matches in three
            dimensions.
          </p>
        </div>
        {error && (
          <div className="error" role="alert">
            {error}
            <button onClick={() => setError("")}>Dismiss</button>
          </div>
        )}
        <div className="workspace">
          <aside>
            <section className="card">
              <div className="step-title">
                <b>01</b>
                <h2>Choose a database</h2>
              </div>
              <label>
                Database
                <select
                  value={database}
                  onChange={(e) => setDatabase(e.target.value)}
                >
                  {dbs.map((d) => (
                    <option value={d.id} key={d.id}>
                      {d.name}
                    </option>
                  ))}
                </select>
              </label>
              {selected && (
                <p className="hint">
                  {selected.description}
                  <br />
                  {selected.molecules?.toLocaleString()} molecules ·{" "}
                  {selected.conformers?.toLocaleString()} conformers
                  {!selected.overlays && (
                    <>
                      <br />
                      Molecule metadata unavailable; overlays are disabled.
                    </>
                  )}
                </p>
              )}
            </section>
            <section className="card">
              <div className="step-title">
                <b>02</b>
                <h2>Prepare your query</h2>
              </div>
              <label className="upload">
                {file ? file.name : "Choose an SDF or MOL file"}
                <input
                  aria-label="Query file"
                  type="file"
                  accept=".sdf,.sd,.mol"
                  onChange={(e) => {
                    setFile(e.target.files?.[0] || null);
                    setSmiles("");
                  }}
                />
              </label>
              <div className="or">or enter SMILES</div>
              <label>
                SMILES
                <input
                  aria-label="SMILES"
                  placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
                  value={smiles}
                  onChange={(e) => {
                    setSmiles(e.target.value);
                    setFile(null);
                  }}
                />
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={generate || !!smiles}
                  disabled={!!smiles}
                  onChange={(e) => setGenerate(e.target.checked)}
                />{" "}
                Generate 3D conformers
              </label>
              {(generate || !!smiles) && (
                <div className="row">
                  <label>
                    Conformers
                    <input
                      type="number"
                      min={1}
                      max={limits.max_conformers}
                      value={count}
                      onChange={(e) => setCount(Number(e.target.value))}
                    />
                  </label>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={minimize}
                      onChange={(e) => setMinimize(e.target.checked)}
                    />
                    Minimize energy
                  </label>
                </div>
              )}
              <button
                className="secondary wide"
                disabled={
                  busy || !!preparing || !!running || (!file && !smiles)
                }
                onClick={() => attempt(prepare)}
              >
                {preparing ? "Preparing…" : "Prepare & preview"}
              </button>
              {job?.preparation && (
                <>
                  <p className="success">
                    {job.preparation.conformers} query conformers ready
                  </p>
                  {job.preparation.warnings.map((w) => (
                    <p className="hint" key={w}>
                      {w}
                    </p>
                  ))}
                </>
              )}
            </section>
            <section className="card">
              <div className="step-title">
                <b>03</b>
                <h2>Run a screen</h2>
              </div>
              <label>
                Scoring mode
                <select value={mode} onChange={(e) => setMode(e.target.value)}>
                  <option value="shape">Shape only</option>
                  {selected?.modes.includes("rescore") && (
                    <option value="rescore">Shape + color rescore</option>
                  )}
                  {selected?.modes.includes("combo") && (
                    <option value="combo">Combo optimization</option>
                  )}
                </select>
              </label>
              <div className="row">
                <label>
                  Top molecules (K)
                  <input
                    type="number"
                    min={1}
                    max={limits.max_k}
                    value={k}
                    onChange={(e) => setK(Number(e.target.value))}
                  />
                </label>
                {mode !== "shape" && (
                  <label>
                    Color weight
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.1}
                      value={mix}
                      onChange={(e) => setMix(Number(e.target.value))}
                    />
                  </label>
                )}
              </div>
              <p className="hint">
                Each molecule is ranked by its best match across query
                conformers.
              </p>
              <button
                className="primary wide"
                disabled={busy || job?.state !== "ready"}
                onClick={() => attempt(screen)}
              >
                Screen database →
              </button>
            </section>
            <section className="card history">
              <h2>Recent jobs</h2>
              <p className="hint">
                Shared lab history · retained {limits.retention_days} days
              </p>
              <button className="quiet" disabled={busy} onClick={() =>
                attempt(async () => setJobs(await api("/jobs")))
              }>Refresh jobs</button>
              {jobs.slice(0, 10).map((j) => (
                <button key={j.id} onClick={() => setJob(j)}>
                  <span>
                    {j.request.database || "Query preparation"}
                    <small>{new Date(j.created * 1000).toLocaleString()}</small>
                  </span>
                  <em>{j.state}</em>
                </button>
              ))}
            </section>
          </aside>
          <div className="results-area">
            {job && (
              <div className="job-bar">
                <div>
                  <strong>{job.state.replace("_", " ")}</strong>
                  <span>
                    Job {job.id.slice(0, 8)} · {Math.round(elapsed)}s elapsed
                    {eta !== null ? ` · ~${eta}s to ${job.progress_basis === "completed" ? "complete" : "submit"} remaining work` : ""}
                  </span>
                  {running && (
                    <>
                      <progress value={job.progress || 0} max={1} />
                      <small>
                        {(job.progress_basis === "completed" ? job.completed_conformers : job.submitted_conformers)?.toLocaleString()} /{" "}
                        {job.total_conformers?.toLocaleString() || "—"}{" "}
                        {job.progress_basis === "completed" ? "conformers completed." : "conformers submitted. Device work may still be in flight."}
                      </small>
                    </>
                  )}
                </div>
                {(running || preparing) && (
                  <button
                    className="quiet"
                    onClick={() =>
                      attempt(async () => {
                        await api("/jobs/" + job.id + "/cancel", {
                          method: "POST",
                        });
                      })
                    }
                  >
                    Cancel
                  </button>
                )}
                {job.error && <p role="alert">{job.error}</p>}
              </div>
            )}
            <Viewer items={overlays.length ? overlays : previewItems} />
            {job?.state === "completed" ? (
              <section className="card hits">
                <div className="panel-heading">
                  <div>
                    <span className="eyebrow">BEST MATCHES</span>
                    <h2>{total.toLocaleString()} molecules</h2>
                  </div>
                  <div className="export">
                    <a href={`/api/jobs/${job.id}/export/csv`}>CSV ↓</a>
                    <a href={`/api/jobs/${job.id}/export/manifest`}>
                      Manifest ↓
                    </a>
                    {overlays.length > 0 && (
                      <a
                        href={`/api/jobs/${job.id}/export/sdf?ranks=${overlays.map((x) => x.row.rank).join(",")}`}
                      >
                        Aligned SDF ↓
                      </a>
                    )}
                  </div>
                </div>
                <div className="table-tools">
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={pin}
                      onChange={(e) => setPin(e.target.checked)}
                    />
                    Pin up to four hits for comparison
                  </label>
                  {overlays.length > 1 && (
                    <button className="quiet" onClick={() => { overlayRequest.current++; setOverlays([]); }}>
                      Clear comparison
                    </button>
                  )}
                </div>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Structure</th>
                        <th>Molecule</th>
                        <th>Shape</th>
                        <th>Color</th>
                        <th>Combo</th>
                        <th>Query</th>
                      </tr>
                    </thead>
                    <tbody>
                      {hits.map((h) => (
                        <tr
                          key={h.rank}
                          className={
                            overlays.some((x) => x.row.rank === h.rank)
                              ? "selected"
                              : ""
                          }
                          tabIndex={0}
                          onClick={() => attempt(() => choose(h.rank))}
                          onKeyDown={(e) => {
                            if (e.key === "Enter")
                              attempt(() => choose(h.rank));
                          }}
                        >
                          <td>{h.rank}</td>
                          <td>
                            <img
                              loading="lazy"
                              alt={"Structure of " + (h.name || h.mol_id)}
                              src={`/api/jobs/${job.id}/hits/${h.rank}/thumbnail`}
                            />
                          </td>
                          <td>
                            <strong>{h.name || h.mol_id}</strong>
                            <small>Conformer {h.conf_idx}</small>
                          </td>
                          <td>{h.shape.toFixed(4)}</td>
                          <td>{h.color.toFixed(4)}</td>
                          <td>{h.combination.toFixed(4)}</td>
                          <td>{h.query_index + 1}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="pagination">
                  <button
                    disabled={page === 0}
                    onClick={() => setPage(page - 1)}
                  >
                    Previous page
                  </button>
                  <span>
                    {page * 50 + 1}–{Math.min((page + 1) * 50, total)} of{" "}
                    {total}
                  </span>
                  <button
                    disabled={(page + 1) * 50 >= total}
                    onClick={() => setPage(page + 1)}
                  >
                    Next page
                  </button>
                </div>
                <div className="pagination">
                  <button
                    disabled={
                      !overlays.length || overlays.at(-1)!.row.rank <= 1
                    }
                    onClick={() =>
                      attempt(() => choose(overlays.at(-1)!.row.rank - 1))
                    }
                  >
                    ← Previous hit
                  </button>
                  <button
                    disabled={
                      !overlays.length || overlays.at(-1)!.row.rank >= total
                    }
                    onClick={() =>
                      attempt(() => choose(overlays.at(-1)!.row.rank + 1))
                    }
                  >
                    Next hit →
                  </button>
                </div>
              </section>
            ) : (
              <div className="empty">
                <h2>Your next lead starts here.</h2>
                <p>
                  Prepare a query on the left to preview its structure.
                  Completed screens appear here as a ranked table and
                  interactive overlays.
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
      <footer>
        ROSHAMBO3 · Exact final scores, optimized molecular poses.
      </footer>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
