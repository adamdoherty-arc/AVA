import { QueryClient } from '@tanstack/react-query';
import { axiosInstance } from './axios';

export const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 1000 * 60 * 5, // 5 minutes
            retry: 1,
            refetchOnWindowFocus: false,
        },
    },
});

// ============ Prefetch Critical Data on App Load ============

// Cache key for localStorage
const WATCHLISTS_CACHE_KEY = 'scanner-watchlists-cache';
const WATCHLISTS_CACHE_EXPIRY = 1000 * 60 * 60; // 1 hour

// Load cached watchlists from localStorage (instant on page load)
function loadCachedWatchlists() {
    try {
        const cached = localStorage.getItem(WATCHLISTS_CACHE_KEY);
        if (cached) {
            const { data, timestamp } = JSON.parse(cached);
            const age = Date.now() - timestamp;
            if (age < WATCHLISTS_CACHE_EXPIRY) {
                return data;
            }
        }
    } catch (e) {
        console.warn('Failed to load cached watchlists:', e);
    }
    return null;
}

// Save watchlists to localStorage
function saveCachedWatchlists(data: unknown) {
    try {
        localStorage.setItem(WATCHLISTS_CACHE_KEY, JSON.stringify({
            data,
            timestamp: Date.now()
        }));
    } catch (e) {
        console.warn('Failed to cache watchlists:', e);
    }
}

// Prefetch watchlists on app load
export async function prefetchCriticalData() {
    // Load from localStorage first (instant)
    const cachedData = loadCachedWatchlists();
    if (cachedData) {
        queryClient.setQueryData(['scanner-watchlists'], cachedData);
    }

    // Then fetch fresh data in background
    try {
        const { data } = await axiosInstance.get('/scanner/watchlists');
        queryClient.setQueryData(['scanner-watchlists'], data);
        saveCachedWatchlists(data);
    } catch (e) {
        console.warn('Failed to prefetch watchlists:', e);
    }
}

// Run prefetch immediately when module loads
prefetchCriticalData();
