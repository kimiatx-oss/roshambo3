import { test, expect } from "@playwright/test";
const mol = `ethanol
     RDKit          3D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
M  END
`;
test("prepare, screen, inspect and pin overlays, export", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  let job: any = null;
  const rows = Array.from({ length: 5 }, (_, i) => ({
    rank: i + 1,
    gid: String(i),
    mol_id: String(i),
    query_index: 0,
    name: `Hit ${i + 1}`,
    shape: 0.95 - i * 0.01,
    color: 0.8,
    combination: 0.87,
  }));
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    let data: any = {};
    if (url.pathname === "/api/databases")
      data = {
        databases: [
          {
            id: "test",
            name: "Test database",
            description: "A curated library",
            molecules: 100,
            conformers: 400,
            modes: ["shape", "rescore", "combo"],
            overlays: true,
          },
        ],
        limits: { max_k: 10000, max_conformers: 50, retention_days: 7 },
      };
    else if (url.pathname === "/api/queries") {
      job = {
        id: "abc123",
        created: Date.now() / 1000,
        state: "ready",
        progress: 0,
        request: {},
        preparation: { conformers: 3, warnings: [], molblock: mol },
      };
      data = job;
    } else if (url.pathname.endsWith("/screen")) {
      job = {
        ...job,
        state: "completed",
        request: { database: "test" },
        progress: 1,
      };
      data = job;
    } else if (url.pathname === "/api/jobs") data = job ? [job] : [];
    else if (url.pathname.endsWith("/hits")) data = { rows, total: 5 };
    else if (url.pathname.endsWith("/overlay")) {
      const rank = Number(url.pathname.split("/").at(-2));
      data = { query: mol, hit: mol, row: rows[rank - 1] };
    } else if (url.pathname.endsWith("/thumbnail")) {
      await route.fulfill({
        contentType: "image/svg+xml",
        body: '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="100"/>',
      });
      return;
    } else data = job;
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await page.getByLabel("SMILES", { exact: true }).fill("CCO");
  await page.getByRole("button", { name: "Prepare & preview" }).click();
  await expect(page.getByText("3 query conformers ready")).toBeVisible();
  await expect(page.locator(".legend")).toContainText("Query");
  await expect(page.getByText(/Query preview · first prepared conformer/)).toBeVisible();
  await page.getByRole("button", { name: "Screen database" }).click();
  await expect(
    page.getByRole("heading", { name: "5 molecules" }),
  ).toBeVisible();
  // Finishing a search must not blank the viewer before a hit is selected.
  await expect(page.locator(".legend")).toContainText("Query");
  await expect(page.getByText(/Query preview · first prepared conformer/)).toBeVisible();
  await expect(page.getByLabel("Representation")).toHaveValue("surface");
  await expect(page.getByTestId("molecular-viewer")).toHaveAttribute("data-surface-state", "ready");
  await expect(page.locator(".legend")).toContainText("gray surface + thin sticks");
  await page.getByRole("row").filter({ hasText: "Hit 1" }).click();
  await expect(page.locator(".legend")).toContainText("Hit 1");
  await expect(page.getByText(/Query preview · first prepared conformer/)).toHaveCount(0);
  await expect(page.getByTestId("molecular-viewer")).toHaveAttribute("data-surface-state", "ready");
  await expect(page.getByLabel("Coloring", { exact: true })).toHaveValue("element");
  await expect(page.getByLabel("Element colors", { exact: true })).toBeVisible();
  await expect(page.locator(".legend")).toContainText("Query");
  await expect(page.locator(".legend")).toContainText("carbon color");
  await page.getByLabel("Pin up to four hits for comparison").check();
  await page.getByRole("row").filter({ hasText: "Hit 2" }).click();
  await expect(page.locator(".legend")).toContainText("Hit 2");
  await expect(page.locator(".legend")).toContainText("Hit 1");
  await expect(page.getByTestId("molecular-viewer")).toHaveAttribute("data-surface-state", "ready");
  await page.getByLabel("Query", { exact: true }).uncheck();
  await expect(page.getByTestId("molecular-viewer")).toHaveAttribute("data-surface-state", "none");
  await page.getByLabel("Query", { exact: true }).check();
  await expect(page.getByTestId("molecular-viewer")).toHaveAttribute("data-surface-state", "ready");
  await expect(page.getByRole("link", { name: "Aligned SDF" })).toHaveAttribute(
    "href",
    /ranks=1,2/,
  );
  await expect(page.getByRole("link", { name: "CSV" })).toHaveAttribute(
    "href",
    /export\/csv/,
  );
  await page.screenshot({
    path: "test-results/results-page.png",
    fullPage: true,
  });
  await page.getByLabel("Representation").selectOption("sphere");
  await expect(page.getByTestId("molecular-viewer")).toHaveAttribute("data-surface-state", "none");
  await expect(page.getByLabel("Element colors", { exact: true })).toBeVisible();
  await page.getByLabel("Coloring", { exact: true }).selectOption("molecule");
  await expect(page.getByLabel("Element colors", { exact: true })).toHaveCount(0);
  await expect(page.locator(".legend")).not.toContainText("carbon color");
  await page.getByLabel("Coloring", { exact: true }).selectOption("element");
  await page.getByLabel("Representation").selectOption("stick");
  await expect(page.getByLabel("Element colors", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Clear comparison" }).click();
  await expect(page.locator(".legend")).toContainText("Query");
  await expect(page.locator(".legend")).not.toContainText("Hit 1");
  await expect(page.getByText(/Query preview · first prepared conformer/)).toBeVisible();
  expect(errors).toEqual([]);
});

test("surface mode preserves query sticks and hit spheres", async ({ page }) => {
  await page.route("**/api/**", (route) => route.fulfill({ json: {
    databases: [], limits: { max_k: 10000, max_conformers: 50, retention_days: 7 },
  } }));
  await page.goto("/");
  const styles = await page.evaluate(async () => {
    const modulePath = "/src/viewer-style.ts";
    const { overlayStyle, QUERY_SURFACE, QUERY_CARBON, HIT_CARBONS } = await import(modulePath);
    return {
      query: overlayStyle("surface", QUERY_CARBON, true),
      hit: overlayStyle("surface", HIT_CARBONS[0], false),
      surface: QUERY_SURFACE,
    };
  });
  expect(styles.query.stick.colorscheme.map.N).toBe(0x3050f8);
  expect(styles.query.sphere).toBeUndefined();
  expect(styles.hit.sphere.colorscheme.map.O).toBe(0xff0d0d);
  expect(styles.hit.stick).toBeUndefined();
  expect(styles.surface.opacity).toBeGreaterThan(0);
  expect(styles.surface.opacity).toBeLessThan(1);
});

test("sticks and spheres change only carbon colors between molecules", async ({ page }) => {
  await page.route("**/api/**", (route) => route.fulfill({ json: {
    databases: [], limits: { max_k: 10000, max_conformers: 50, retention_days: 7 },
  } }));
  await page.goto("/");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/viewer-style.ts";
    const { overlayStyle, QUERY_CARBON, HIT_CARBONS, ELEMENT_COLORS } = await import(modulePath);
    const before = { ...ELEMENT_COLORS };
    const styles = ["stick", "sphere"].map((representation) => {
      const query = overlayStyle(representation, QUERY_CARBON, true)[representation];
      const hit = overlayStyle(representation, HIT_CARBONS[0], false)[representation];
      const second = overlayStyle(representation, HIT_CARBONS[1], false)[representation];
      const uniform = overlayStyle(representation, HIT_CARBONS[0], false, false)[representation];
      return { query, hit, second, uniform };
    });
    return { styles, before, after: { ...ELEMENT_COLORS }, carbon: QUERY_CARBON, hitCarbon: HIT_CARBONS[0] };
  });
  expect(result.after).toEqual(result.before);
  for (const { query, hit, second, uniform } of result.styles) {
    expect(query.colorscheme.prop).toBe("elem");
    expect(query.colorscheme.map.C).toBe(result.carbon);
    expect(hit.colorscheme.map.C).toBe(result.hitCarbon);
    expect(second.colorscheme.map.C).not.toBe(hit.colorscheme.map.C);
    for (const element of Object.keys(result.before).filter((key) => key !== "C")) {
      expect(query.colorscheme.map[element]).toBe(hit.colorscheme.map[element]);
      expect(second.colorscheme.map[element]).toBe(hit.colorscheme.map[element]);
    }
    expect(hit.colorscheme.map.N).toBe(0x3050f8);
    expect(hit.colorscheme.map.O).toBe(0xff0d0d);
    expect(hit.colorscheme.map.S).toBe(0xffff30);
    expect(uniform.color).toBe(result.hitCarbon);
    expect(uniform.colorscheme).toBeUndefined();
  }
  expect(result.styles[0].query.radius).toBeLessThan(result.styles[0].hit.radius);
});
