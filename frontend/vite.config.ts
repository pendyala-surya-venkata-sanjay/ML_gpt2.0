import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

// Force Vite to use the `frontend/` directory as the project root.
// This prevents dev-server 404s if the process working directory differs.
const rootDir = fileURLToPath(new URL('.', import.meta.url))

export default defineConfig({
  root: rootDir,
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
  },
})

