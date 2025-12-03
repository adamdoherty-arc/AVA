/**
 * Centralized API Configuration
 *
 * Update this single file to change the backend URL for the entire frontend.
 * All API calls should import from here.
 */

// Backend API base URL - SINGLE SOURCE OF TRUTH
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8002/api';

// Base URL without /api suffix (for pages that append their own paths)
export const API_HOST = import.meta.env.VITE_API_URL?.replace('/api', '') || 'http://localhost:8002';
