import { test, expect, type Route } from "@playwright/test";

const mol = `ethanol
     RDKit          3D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  END
`;
const record = (id: string, state = "completed", submitted_conformers = 0) => ({
  id, state, created: Date.now() / 1000, progress: 0.1, submitted_conformers,
  total_conformers: 100, request: { database: id },
  preparation: { conformers: 1, warnings: [], molblock: mol },
});
const hit = (name: string) => ({ rank: 1, gid: "0", mol_id: "1", conf_idx: 0, query_index: 0,
  name, shape: 0.8, color: 0.8, combination: 0.8 });
const databases = { databases: [{ id: "A", name: "A", modes: ["shape"], overlays: true }],
  limits: { max_k: 10000, max_conformers: 50, retention_days: 7 } };

for (const delayed of ["overlay", "hits"] as const) {
  test(`late ${delayed} response cannot cross job identity`, async ({ page }) => {
    let pending: Route | undefined;
    const errors: string[] = [];
    page.on("pageerror", error => errors.push(error.message));
    await page.route("**/api/**", async route => {
      const url = new URL(route.request().url());
      const path = url.pathname;
      if (path === "/api/databases") return route.fulfill({ json: databases });
      if (path === "/api/jobs") return route.fulfill({ json: [record("A"), record("B")] });
      if (path === "/api/jobs/A") return route.fulfill({ json: record("A") });
      if (path.endsWith("/thumbnail")) return route.fulfill({ contentType: "image/svg+xml", body: "<svg/>" });
      if (path === `/api/jobs/A/${delayed === "hits" ? "hits" : "hits/1/overlay"}`) {
        pending = route;
        return;
      }
      if (path.endsWith("/hits")) return route.fulfill({ json: { rows: [hit(path.includes("/A/") ? "A-hit" : "B-hit")], total: 1 } });
      return route.fulfill({ json: {} });
    });
    await page.goto("/?job=A");
    if (delayed === "overlay") {
      await page.getByRole("row").filter({ hasText: "A-hit" }).click();
    }
    await expect.poll(() => !!pending).toBe(true);
    await page.locator(".history button").filter({ hasText: /^B/ }).click();
    await expect(page.getByRole("row").filter({ hasText: "B-hit" })).toBeVisible();
    await pending!.fulfill({ json: delayed === "hits"
      ? { rows: [hit("A-hit")], total: 999 }
      : { query: mol, hit: mol, row: hit("A-hit") } });
    // Let the fulfilled response and React effects settle without wall-clock sleeps.
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    await expect(page.getByRole("row").filter({ hasText: "B-hit" })).toBeVisible();
    await expect(page.getByRole("row").filter({ hasText: "A-hit" })).toHaveCount(0);
    await expect(page.locator(".legend")).not.toContainText("A-hit");
    await expect(page.getByRole("link", { name: "Aligned SDF" })).toHaveCount(0);
    expect(errors).toEqual([]);
  });
}

test("progress and form updates do not rebuild the query surface", async ({ page }) => {
  await page.clock.install();
  let submitted = 1;
  await page.route("**/api/**", async route => {
    const path = new URL(route.request().url()).pathname;
    await route.fulfill({ json: path === "/api/databases" ? databases
      : path === "/api/jobs" ? [record("A", "screening", submitted)]
      : record("A", "screening", submitted) });
  });
  await page.goto("/?job=A");
  const viewer = page.getByTestId("molecular-viewer");
  await expect(viewer).toHaveAttribute("data-surface-state", "ready");
  await page.evaluate(() => {
    const target = document.querySelector('[data-testid="molecular-viewer"]')!;
    (window as any).surfaceChanges = 0;
    new MutationObserver(records => { (window as any).surfaceChanges += records.length; })
      .observe(target, { attributes: true, attributeFilter: ["data-surface-state"] });
  });
  await page.getByLabel("SMILES", { exact: true }).fill("CCN");
  submitted = 25;
  await page.clock.runFor(1600);
  await expect(page.locator(".job-bar")).toContainText("25 / 100");
  await expect(viewer).toHaveAttribute("data-surface-state", "ready");
  expect(await page.evaluate(() => (window as any).surfaceChanges)).toBe(0);
});

test("ETA uses the backend submission estimate only during screening", async ({ page }) => {
  await page.clock.install();
  let state = "waiting_gpu";
  let eta: number | null = 42.1;
  const job = () => ({ ...record("A", state), created: Date.now() / 1000 - 3600, eta_seconds: eta });
  await page.route("**/api/**", async route => {
    const path = new URL(route.request().url()).pathname;
    await route.fulfill({ json: path === "/api/databases" ? databases
      : path === "/api/jobs" ? [job()]
      : path.endsWith("/hits") ? { rows: [], total: 0 } : job() });
  });
  await page.goto("/?job=A");
  const bar = page.locator(".job-bar");
  await expect(bar).toContainText("waiting gpu");
  await expect(bar).not.toContainText("to submit remaining work");
  state = "screening";
  await page.clock.runFor(1600);
  await expect(bar).toContainText("~43s to submit remaining work");
  eta = null;
  await page.clock.runFor(1600);
  await expect(bar).not.toContainText("to submit remaining work");
  eta = 42.1;
  state = "finalizing";
  await page.clock.runFor(1600);
  await expect(bar).toContainText("finalizing");
  await expect(bar).not.toContainText("to submit remaining work");
  state = "completed";
  await page.clock.runFor(1600);
  await expect(bar).toContainText("completed");
  await expect(bar).not.toContainText("to submit remaining work");
});

test("CPU jobs show completed work and remain cancellable while waiting", async ({ page }) => {
  await page.clock.install();
  let state = "waiting_cpu";
  const job = () => ({ ...record("A", state), progress_basis: "completed",
    completed_conformers: 25, eta_seconds: 12 });
  await page.route("**/api/**", async route => {
    const path = new URL(route.request().url()).pathname;
    await route.fulfill({ json: path === "/api/databases" ? databases
      : path === "/api/jobs" ? [job()] : job() });
  });
  await page.goto("/?job=A");
  await expect(page.locator(".job-bar")).toContainText("waiting cpu");
  await expect(page.getByRole("button", { name: "Cancel", exact: true })).toBeVisible();
  state = "screening";
  await page.clock.runFor(1600);
  await expect(page.locator(".job-bar")).toContainText("25 / 100 conformers completed.");
  await expect(page.locator(".job-bar")).toContainText("~12s to complete remaining work");
  await expect(page.locator(".job-bar")).not.toContainText("Device work may still be in flight");
});
