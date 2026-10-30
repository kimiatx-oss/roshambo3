import { defineConfig } from "@playwright/test";
const port = Number(process.env.ROSHAMBO3_WEB_TEST_PORT || 5173);
export default defineConfig({
  testDir: "./tests",
  // Hosted CI runners render WebGL in software on shared cores; a retry is
  // reported as flaky rather than hidden, and keeps a trace of the first failure.
  retries: process.env.CI ? 2 : 0,
  use: { baseURL: `http://127.0.0.1:${port}`, headless: true, trace: "retain-on-first-failure" },
  webServer: {
    command: `npm run dev -- --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: false,
  },
});
