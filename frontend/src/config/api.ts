/**
 * Centralized API Configuration
 *
 * HARDCODED PORTS - DO NOT CHANGE WITHOUT UPDATING ALL CONFIGS:
 * - Backend: port 8002
 * - Frontend: port 5181
 *
 * Uses relative URLs to go through Vite proxy (avoids CORS issues in dev)
 * All API calls should import from here.
 */

// Backend server URL - HARDCODED
export const BACKEND_URL = 'http://localhost:8002';

// WebSocket server URL - HARDCODED
export const WS_URL = 'ws://localhost:8002';

// Backend API base URL - Relative path goes through Vite proxy to port 8002
export const API_BASE_URL = '/api';

// Full API URL for direct calls (fetch, SSE, etc)
export const API_FULL_URL = `${BACKEND_URL}/api`;

// Full WebSocket API URL
export const WS_API_URL = `${WS_URL}/api`;

// Base URL without /api suffix (for pages that append their own paths)
export const API_HOST = BACKEND_URL;
