import { defineConfig } from "@playwright/test";
const port = Number(process.env.ROSHAMBO3_WEB_TEST_PORT || 5173);
export default defineConfig({
  testDir: "./tests",
  use: { baseURL: `http://127.0.0.1:${port}`, headless: true },
  webServer: {
    command: `npm run dev -- --port ${port} --strictPort`,
    url: `http://127.0.0.1:${port}`,
    reuseExistingServer: false,
  },
});
