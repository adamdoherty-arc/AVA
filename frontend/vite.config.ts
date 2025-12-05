/// <reference types="vitest" />
import path from "path"
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  // Test configuration
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: ['node_modules/', 'src/setupTests.ts'],
    },
  },
  server: {
    port: 5181,  // HARDCODED - Do not change
    strictPort: true,  // Fail if port is already in use
    proxy: {
      '/api': {
        target: 'http://localhost:8002',  // HARDCODED - Backend runs on 8002
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8002',  // HARDCODED - WebSocket proxy
        ws: true,
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },

  // Production build optimizations
  build: {
    target: 'esnext',
    minify: 'esbuild',  // Using esbuild for faster builds (default)
    sourcemap: false,
    cssCodeSplit: true,
    chunkSizeWarningLimit: 500,

    // Note: For Vite 6+, terser must be installed separately: npm i -D terser
    // Using esbuild minification (default) for faster builds

    rollupOptions: {
      output: {
        // Manual chunk splitting for optimal caching
        manualChunks: {
          // React core
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],

          // Data management
          'data-vendor': ['@tanstack/react-query', 'zustand', 'axios'],

          // UI framework
          'ui-vendor': ['framer-motion', 'lucide-react', 'clsx'],

          // Charts (large bundle - lazy load when needed)
          'charts-vendor': ['recharts'],
        },

        // Asset naming for better caching
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name?.split('.') || [];
          const ext = info[info.length - 1];
          if (/png|jpe?g|svg|gif|tiff|bmp|ico/i.test(ext)) {
            return 'assets/images/[name]-[hash][extname]';
          }
          if (/css/i.test(ext)) {
            return 'assets/css/[name]-[hash][extname]';
          }
          return 'assets/[name]-[hash][extname]';
        },
        chunkFileNames: 'assets/js/[name]-[hash].js',
        entryFileNames: 'assets/js/[name]-[hash].js',
      },
    },
  },

  // Dependency pre-bundling optimization
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@tanstack/react-query',
      'zustand',
      'axios',
      'framer-motion',
      'lucide-react',
      'clsx',
      'sonner',
    ],
    exclude: ['@tanstack/react-query-devtools'],
  },

  // Enable faster builds
  esbuild: {
    legalComments: 'none',
    treeShaking: true,
  },
})
