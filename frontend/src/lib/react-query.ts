import { QueryClient } from '@tanstack/react-query';
import { axiosInstance } from './axios';

// =============================================================================
// Query Client Configuration
// =============================================================================

export const queryClient = new QueryClient({
    defaultOptions: {
        queries: {
            staleTime: 1000 * 60 * 5, // 5 minutes
            retry: 1,
            refetchOnWindowFocus: false,
            // Network mode: only fetch when online
            networkMode: 'online',
        },
        mutations: {
            // Retry mutations once on failure
            retry: 1,
            retryDelay: 1000,
        },
    },
});

// =============================================================================
// Smart Cache with localStorage Persistence
// =============================================================================

interface CacheEntry<T> {
    data: T;
    timestamp: number;
}

const CACHE_PREFIX = 'magnus-cache:';
const DEFAULT_CACHE_EXPIRY = 1000 * 60 * 60; // 1 hour

/**
 * Load cached data from localStorage
 */
function loadFromLocalStorage<T>(key: string, maxAge: number = DEFAULT_CACHE_EXPIRY): T | null {
    try {
        const cached = localStorage.getItem(CACHE_PREFIX + key);
        if (cached) {
            const { data, timestamp }: CacheEntry<T> = JSON.parse(cached);
            const age = Date.now() - timestamp;
            if (age < maxAge) {
                return data;
            }
            // Clean up expired cache
            localStorage.removeItem(CACHE_PREFIX + key);
        }
    } catch (e) {
        console.warn(`Failed to load cached ${key}:`, e);
    }
    return null;
}

/**
 * Save data to localStorage cache
 */
function saveToLocalStorage<T>(key: string, data: T): void {
    try {
        const entry: CacheEntry<T> = {
            data,
            timestamp: Date.now(),
        };
        localStorage.setItem(CACHE_PREFIX + key, JSON.stringify(entry));
    } catch (e) {
        console.warn(`Failed to cache ${key}:`, e);
    }
}

/**
 * Clear all Magnus cache entries from localStorage
 */
export function clearLocalCache(): void {
    const keys = Object.keys(localStorage).filter(k => k.startsWith(CACHE_PREFIX));
    keys.forEach(k => localStorage.removeItem(k));
    console.debug(`[Cache] Cleared ${keys.length} cached entries`);
}

// =============================================================================
// Smart Prefetch Configuration
// =============================================================================

interface PrefetchConfig {
    queryKey: string[];
    endpoint: string;
    localStorageKey: string;
    cacheExpiry?: number;
    critical?: boolean; // If true, errors are logged as warnings
}

const PREFETCH_CONFIG: PrefetchConfig[] = [
    {
        queryKey: ['scanner-watchlists'],
        endpoint: '/scanner/watchlists',
        localStorageKey: 'watchlists',
        cacheExpiry: 1000 * 60 * 60, // 1 hour
        critical: false, // Has fallback presets
    },
    {
        queryKey: ['health'],
        endpoint: '/health',
        localStorageKey: 'health',
        cacheExpiry: 1000 * 60 * 5, // 5 minutes
        critical: false,
    },
];

// =============================================================================
// Prefetch Functions
// =============================================================================

/**
 * Prefetch a single endpoint with localStorage fallback
 */
async function prefetchEndpoint(config: PrefetchConfig): Promise<void> {
    const { queryKey, endpoint, localStorageKey, cacheExpiry, critical } = config;

    // Load from localStorage first (instant)
    const cachedData = loadFromLocalStorage(localStorageKey, cacheExpiry);
    if (cachedData) {
        queryClient.setQueryData(queryKey, cachedData);
        console.debug(`[Prefetch] Loaded ${localStorageKey} from localStorage cache`);
    }

    // Then fetch fresh data in background (non-blocking)
    try {
        const { data } = await axiosInstance.get(endpoint);
        queryClient.setQueryData(queryKey, data);
        saveToLocalStorage(localStorageKey, data);
        console.debug(`[Prefetch] Updated ${localStorageKey} from API`);
    } catch (e) {
        const logFn = critical ? console.warn : console.debug;
        logFn(`[Prefetch] Failed to fetch ${endpoint}:`, e);
    }
}

/**
 * Prefetch all critical data on app load
 */
export async function prefetchCriticalData(): Promise<void> {
    console.debug('[Prefetch] Starting critical data prefetch...');

    // Run all prefetches in parallel
    await Promise.allSettled(
        PREFETCH_CONFIG.map(config => prefetchEndpoint(config))
    );

    console.debug('[Prefetch] Critical data prefetch complete');
}

// =============================================================================
// Smart Cache Warming
// =============================================================================

/**
 * Warm cache for a specific route before navigation
 */
export async function warmCacheForRoute(route: string): Promise<void> {
    const routePrefetchMap: Record<string, () => Promise<void>> = {
        '/positions': async () => {
            // Pre-warm positions data
            await queryClient.prefetchQuery({
                queryKey: ['positions'],
                queryFn: () => axiosInstance.get('/portfolio/positions').then(r => r.data),
                staleTime: 30000,
            });
        },
        '/scanner': async () => {
            // Pre-warm scanner data
            await queryClient.prefetchQuery({
                queryKey: ['scanner-watchlists'],
                queryFn: () => axiosInstance.get('/scanner/watchlists').then(r => r.data),
                staleTime: 60000,
            });
        },
        '/dashboard': async () => {
            // Pre-warm dashboard data
            await queryClient.prefetchQuery({
                queryKey: ['dashboard-summary'],
                queryFn: () => axiosInstance.get('/dashboard/summary').then(r => r.data),
                staleTime: 30000,
            });
        },
        '/sports': async () => {
            // Pre-warm sports data
            await queryClient.prefetchQuery({
                queryKey: ['live-games'],
                queryFn: () => axiosInstance.get('/sports/live').then(r => r.data),
                staleTime: 30000,
            });
        },
    };

    const prefetchFn = routePrefetchMap[route];
    if (prefetchFn) {
        try {
            await prefetchFn();
            console.debug(`[Cache] Warmed cache for route: ${route}`);
        } catch (e) {
            console.debug(`[Cache] Failed to warm cache for ${route}:`, e);
        }
    }
}

/**
 * Invalidate and refetch cache for a category
 */
export async function invalidateCategory(category: string): Promise<void> {
    const categoryQueryKeys: Record<string, string[][]> = {
        portfolio: [['positions'], ['portfolio-summary'], ['enriched-positions']],
        scanner: [['scanner-watchlists'], ['scan-history'], ['stored-premiums']],
        sports: [['live-games'], ['upcoming-games'], ['best-bets']],
        ai: [['ai-anomalies'], ['ai-risk-score'], ['ai-recommendations']],
    };

    const queryKeys = categoryQueryKeys[category];
    if (queryKeys) {
        await Promise.all(
            queryKeys.map(key =>
                queryClient.invalidateQueries({ queryKey: key })
            )
        );
        console.debug(`[Cache] Invalidated category: ${category}`);
    }
}

// =============================================================================
// Visibility-Based Refetch
// =============================================================================

let visibilityRefetchEnabled = true;

/**
 * Enable/disable visibility-based refetching
 */
export function setVisibilityRefetch(enabled: boolean): void {
    visibilityRefetchEnabled = enabled;
}

// Refetch stale queries when tab becomes visible
if (typeof document !== 'undefined') {
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible' && visibilityRefetchEnabled) {
            // Only refetch if data is stale (older than 2 minutes)
            const staleTime = 2 * 60 * 1000;
            queryClient.refetchQueries({
                predicate: (query) => {
                    const state = query.state;
                    if (!state.dataUpdatedAt) return false;
                    return Date.now() - state.dataUpdatedAt > staleTime;
                },
            });
        }
    });
}

// =============================================================================
// Initialize
// =============================================================================

// Run prefetch when module loads (deferred to avoid blocking)
if (typeof window !== 'undefined') {
    // Use requestIdleCallback if available, otherwise setTimeout
    const scheduleInit = window.requestIdleCallback || ((fn) => setTimeout(fn, 100));
    scheduleInit(() => {
        prefetchCriticalData();
    });
}
