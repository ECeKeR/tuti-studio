import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: `http://localhost:${process.env.VITE_PYTHON_PORT || 5050}`,
        changeOrigin: true
      }
    }
  }
});
