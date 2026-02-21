import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
    plugins: [react()],
    build: {
        outDir: "../backend/static",
        emptyOutDir: true,
        sourcemap: true,
        target: "esnext"
    },
    server: {
        proxy: {
            "/query": "http://localhost:8000",
            "/query_stream": "http://localhost:8000"
        }
    }
});
