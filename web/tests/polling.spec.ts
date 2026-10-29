import { test, expect } from "@playwright/test";

for (const finalState of ["ready", "completed", "failed", "canceled"]) {
  test(`polling stops at ${finalState} and resumes for new work`, async ({ page }) => {
    await page.clock.install();
    let state = "preparing";
    let requests = 0;
    const record = () => ({
      id: "poll-test", state, created: Date.now() / 1000,
      progress: 0, request: {}, submitted_conformers: 0,
      preparation: { conformers: 1, warnings: [], molblock: "" },
    });
    await page.route("**/api/**", async (route) => {
      const path = new URL(route.request().url()).pathname;
      let data: any;
      if (path === "/api/databases") {
        data = { databases: [], limits: { max_k: 10000, max_conformers: 50, retention_days: 7 } };
      } else if (path.endsWith("/hits")) {
        data = { rows: [], total: 0 };
      } else {
        requests++;
        data = path === "/api/jobs" ? [record()] : record();
      }
      await route.fulfill({ json: data });
    });
    await page.goto("/?job=poll-test");
    await expect(page.locator(".job-bar strong")).toHaveText("preparing");
    // Known active jobs continue to update, including transitions under the
    // same ID (preparation -> ready -> queued -> completed).
    state = finalState;
    await page.clock.runFor(1600);
    await expect(page.locator(".job-bar strong")).toHaveText(finalState);
    // Let any final history refresh settle, then advance several poll periods.
    await page.clock.runFor(1600);
    const settled = requests;
    await page.clock.runFor(6000);
    expect(requests).toBe(settled);

    // A manual history refresh discovers new work and restarts polling.
    state = "queued";
    await page.getByRole("button", { name: "Refresh jobs" }).click();
    await page.getByRole("button", { name: /Query preparation.*queued/ }).click();
    await expect(page.locator(".job-bar strong")).toHaveText("queued");
    await expect(page.locator(".legend")).toContainText("Query");
    state = "completed";
    await page.clock.runFor(1600);
    await expect(page.locator(".job-bar strong")).toHaveText("completed");
    await expect(page.locator(".legend")).toContainText("Query");
    await page.clock.runFor(1600);
    const completed = requests;
    await page.clock.runFor(6000);
    expect(requests).toBe(completed);
  });
}

test("an empty idle history does not poll", async ({ page }) => {
  await page.clock.install();
  let requests = 0;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (path === "/api/jobs") requests++;
    await route.fulfill({ json: path === "/api/jobs" ? [] : {
      databases: [], limits: { max_k: 10000, max_conformers: 50, retention_days: 7 },
    } });
  });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Refresh jobs" })).toBeVisible();
  await page.clock.runFor(1600);
  const settled = requests;
  await page.clock.runFor(6000);
  expect(requests).toBe(settled);
});
