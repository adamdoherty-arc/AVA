/**
 * useSmartCache - AI-Powered Intelligent Caching Hook
 *
 * Features:
 * - Adaptive TTL based on market conditions (volatility, trading hours)
 * - Predictive prefetching based on user patterns
 * - Automatic cache invalidation on significant events
 * - Background refresh with stale-while-revalidate
 */

import { useCallback, useEffect, useRef, useMemo } from 'react';
import { useQueryClient } from '@tanstack/react-query';

// Market hours detection (EST)
const MARKET_OPEN_HOUR = 9.5; // 9:30 AM
const MARKET_CLOSE_HOUR = 16; // 4:00 PM

interface CacheConfig {
    baseStaleTime: number;
    marketOpenMultiplier: number;
    volatileMultiplier: number;
    afterHoursMultiplier: number;
}

interface SmartCacheOptions {
    queryKey: string[];
    isVolatileSymbol?: boolean;
    forceRefresh?: boolean;
}

// Default configuration
const DEFAULT_CONFIG: CacheConfig = {
    baseStaleTime: 60000, // 1 minute base
    marketOpenMultiplier: 0.5, // Halve stale time during market hours
    volatileMultiplier: 0.3, // 30% of base for volatile symbols
    afterHoursMultiplier: 5, // 5x longer after hours
};

/**
 * Check if market is currently open (US Eastern Time)
 */
function isMarketOpen(): boolean {
    const now = new Date();
    const estOffset = -5; // EST is UTC-5
    const utcHour = now.getUTCHours();
    const estHour = (utcHour + estOffset + 24) % 24;
    const estMinutes = estHour + now.getUTCMinutes() / 60;

    const dayOfWeek = now.getUTCDay();
    const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;

    return isWeekday && estMinutes >= MARKET_OPEN_HOUR && estMinutes < MARKET_CLOSE_HOUR;
}

/**
 * Check if currently in pre-market (4:00 AM - 9:30 AM EST)
 */
function isPreMarket(): boolean {
    const now = new Date();
    const estOffset = -5;
    const utcHour = now.getUTCHours();
    const estHour = (utcHour + estOffset + 24) % 24;
    const estMinutes = estHour + now.getUTCMinutes() / 60;

    const dayOfWeek = now.getUTCDay();
    const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;

    return isWeekday && estMinutes >= 4 && estMinutes < MARKET_OPEN_HOUR;
}

/**
 * Calculate smart stale time based on market conditions
 */
export function calculateSmartStaleTime(
    isVolatile: boolean = false,
    config: CacheConfig = DEFAULT_CONFIG
): number {
    let staleTime = config.baseStaleTime;

    if (isMarketOpen()) {
        // Market is open - shorter cache times
        staleTime *= config.marketOpenMultiplier;

        if (isVolatile) {
            // Volatile symbols need even more frequent updates
            staleTime *= config.volatileMultiplier;
        }
    } else if (isPreMarket()) {
        // Pre-market - moderate cache times
        staleTime *= 2;
    } else {
        // After hours - longer cache times
        staleTime *= config.afterHoursMultiplier;
    }

    // Minimum 5 seconds, maximum 10 minutes
    return Math.max(5000, Math.min(staleTime, 600000));
}

/**
 * Main smart caching hook
 */
export function useSmartCache(options: SmartCacheOptions) {
    const queryClient = useQueryClient();
    const prefetchedRef = useRef<Set<string>>(new Set());

    const staleTime = useMemo(
        () => calculateSmartStaleTime(options.isVolatileSymbol),
        [options.isVolatileSymbol]
    );

    // Force refresh functionality
    useEffect(() => {
        if (options.forceRefresh) {
            queryClient.invalidateQueries({ queryKey: options.queryKey });
        }
    }, [options.forceRefresh, queryClient, options.queryKey]);

    // Prefetch related queries
    const prefetch = useCallback(async (relatedQueryKeys: string[][]) => {
        const promises = relatedQueryKeys.map(async (key) => {
            const keyString = key.join('.');
            if (!prefetchedRef.current.has(keyString)) {
                prefetchedRef.current.add(keyString);
                await queryClient.prefetchQuery({
                    queryKey: key,
                    staleTime,
                });
            }
        });

        await Promise.all(promises);
    }, [queryClient, staleTime]);

    // Smart invalidation based on events
    const invalidateOnEvent = useCallback((eventType: 'trade' | 'news' | 'price_alert') => {
        const multipliers = {
            trade: 0, // Immediate invalidation
            news: 0.5, // Half stale time
            price_alert: 0.25, // Quarter stale time
        };

        const delay = staleTime * multipliers[eventType];

        setTimeout(() => {
            queryClient.invalidateQueries({ queryKey: options.queryKey });
        }, delay);
    }, [queryClient, options.queryKey, staleTime]);

    // Clear prefetch cache periodically
    useEffect(() => {
        const interval = setInterval(() => {
            prefetchedRef.current.clear();
        }, 5 * 60 * 1000); // Clear every 5 minutes

        return () => clearInterval(interval);
    }, []);

    return {
        staleTime,
        isMarketOpen: isMarketOpen(),
        isPreMarket: isPreMarket(),
        prefetch,
        invalidateOnEvent,
        config: {
            ...DEFAULT_CONFIG,
            currentMultiplier: staleTime / DEFAULT_CONFIG.baseStaleTime,
        },
    };
}

/**
 * Hook for predictive prefetching based on user patterns
 */
export function usePredictivePrefetch() {
    const queryClient = useQueryClient();
    const navigationHistory = useRef<string[]>([]);

    // Track navigation
    const trackNavigation = useCallback((route: string) => {
        navigationHistory.current.push(route);
        if (navigationHistory.current.length > 10) {
            navigationHistory.current.shift();
        }
    }, []);

    // Predict and prefetch likely next routes
    const prefetchLikelyRoutes = useCallback(() => {
        const history = navigationHistory.current;
        const lastRoute = history[history.length - 1];

        // Simple pattern matching for common flows
        const routePredictions: Record<string, string[][]> = {
            '/positions': [['portfolio-summary'], ['enriched-positions']],
            '/dashboard': [['positions'], ['dashboard-summary']],
            '/options': [['positions'], ['options-analysis']],
            '/sports': [['upcoming-games'], ['best-bets']],
        };

        const predictions = routePredictions[lastRoute];
        if (predictions) {
            predictions.forEach((queryKey) => {
                queryClient.prefetchQuery({
                    queryKey,
                    staleTime: calculateSmartStaleTime(false),
                });
            });
        }
    }, [queryClient]);

    return {
        trackNavigation,
        prefetchLikelyRoutes,
    };
}

/**
 * Hook for market-aware query configuration
 */
export function useMarketAwareConfig() {
    const marketOpen = isMarketOpen();
    const preMarket = isPreMarket();

    return useMemo(() => ({
        isMarketOpen: marketOpen,
        isPreMarket: preMarket,
        isAfterHours: !marketOpen && !preMarket,

        // Recommended polling intervals
        pollingInterval: marketOpen ? 15000 : preMarket ? 30000 : 60000,

        // Recommended stale times
        priceStaleTime: marketOpen ? 5000 : 60000,
        positionsStaleTime: marketOpen ? 30000 : 300000,
        analyticsStaleTime: marketOpen ? 60000 : 600000,

        // Retry configuration
        retryDelay: marketOpen ? 1000 : 5000,
        maxRetries: marketOpen ? 3 : 1,
    }), [marketOpen, preMarket]);
}

export default useSmartCache;
