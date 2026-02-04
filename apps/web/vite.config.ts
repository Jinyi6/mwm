import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ['antd', '@ant-design/plots'],
  },
  build: {
    target: 'es2019',
    sourcemap: false,
    cssCodeSplit: true,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ['react', 'react-dom'],
          antd: ['antd'],
          plots: ['@ant-design/plots'],
        },
      },
    },
  },
  server: {
    port: 5173,
  },
})
