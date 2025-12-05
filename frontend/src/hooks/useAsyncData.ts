/**
 * Modern Async Data Hook
 * =======================
 *
 * Production-grade React hook for async data fetching with:
 * - Automatic loading states
 * - Error handling and retry
 * - Stale-while-revalidate
 * - Optimistic updates
 * - Cache invalidation
 * - Real-time updates via SSE
 *
 * @author AVA Trading Platform
 * @updated 2025-11-29
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { apiClient, isAPIError, formatAPIError } from '@/lib/api-client';
import type { APIError } from '@/lib/api-client';

// =============================================================================
// Types
// =============================================================================

export interface AsyncState<T> {
  data: T | null;
  isLoading: boolean;
  isValidating: boolean;
  isError: boolean;
  error: string | null;
  errorDetails: APIError | null;
  isStale: boolean;
  lastUpdated: Date | null;
}

export interface AsyncActions<T> {
  refetch: () => Promise<void>;
  mutate: (data: T | ((prev: T | null) => T)) => void;
  reset: () => void;
  invalidate: () => void;
}

export interface UseAsyncDataOptions<T> {
  /** Initial data value */
  initialData?: T | null;
  /** Enable automatic fetching */
  enabled?: boolean;
  /** Refetch interval in ms (0 = disabled) */
  refetchInterval?: number;
  /** Stale time in ms before revalidation */
  staleTime?: number;
  /** Retry count on failure */
  retryCount?: number;
  /** Retry delay in ms */
  retryDelay?: number;
  /** Transform response data */
  transform?: (data: unknown) => T;
  /** Callback on success */
  onSuccess?: (data: T) => void;
  /** Callback on error */
  onError?: (error: APIError) => void;
  /** Cache key for deduplication */
  cacheKey?: string;
}

export type UseAsyncDataResult<T> = AsyncState<T> & AsyncActions<T>;

// =============================================================================
// Main Hook
// =============================================================================

/**
 * Hook for fetching and managing async data.
 *
 * @example
 * ```tsx
 * const { data, isLoading, error, refetch } = useAsyncData<Position[]>(
 *   '/portfolio/positions',
 *   {
 *     refetchInterval: 30000,
 *     onSuccess: (positions) => console.log('Loaded', positions.length),
 *   }
 * );
 * ```
 */
export function useAsyncData<T>(
  url: string | null,
  options: UseAsyncDataOptions<T> = {}
): UseAsyncDataResult<T> {
  const {
    initialData = null,
    enabled = true,
    refetchInterval = 0,
    staleTime = 30000,
    retryCount = 3,
    retryDelay = 1000,
    transform,
    onSuccess,
    onError,
    cacheKey,
  } = options;

  // State
  const [state, setState] = useState<AsyncState<T>>({
    data: initialData,
    isLoading: enabled && url !== null,
    isValidating: false,
    isError: false,
    error: null,
    errorDetails: null,
    isStale: true,
    lastUpdated: null,
  });

  // Refs for cleanup and tracking
  const mountedRef = useRef(true);
  const abortControllerRef = useRef<AbortController | null>(null);
  const retryCountRef = useRef(0);

  // Query client for cache invalidation
  const queryClient = useQueryClient();

  // Fetch function
  const fetchData = useCallback(
    async (isRevalidation = false) => {
      if (!url || !enabled) return;

      // Cancel any existing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      // Update loading state
      setState((prev) => ({
        ...prev,
        isLoading: !isRevalidation && !prev.data,
        isValidating: isRevalidation || !!prev.data,
        isError: false,
        error: null,
        errorDetails: null,
      }));

      try {
        const response = await apiClient.get<T>(url, {
          signal: abortControllerRef.current.signal,
          skipCache: isRevalidation,
        });

        // Transform if needed
        const data = transform ? transform(response) : response;

        if (mountedRef.current) {
          setState({
            data,
            isLoading: false,
            isValidating: false,
            isError: false,
            error: null,
            errorDetails: null,
            isStale: false,
            lastUpdated: new Date(),
          });

          retryCountRef.current = 0;
          onSuccess?.(data);
        }
      } catch (err) {
        // Ignore abort errors
        if (err instanceof Error && err.name === 'AbortError') {
          return;
        }

        const apiError: APIError = isAPIError(err)
          ? err
          : {
              code: 500,
              name: 'UNKNOWN_ERROR',
              message: formatAPIError(err),
              timestamp: new Date().toISOString(),
            };

        // Retry logic
        if (retryCountRef.current < retryCount) {
          retryCountRef.current++;
          setTimeout(() => fetchData(isRevalidation), retryDelay * retryCountRef.current);
          return;
        }

        if (mountedRef.current) {
          setState((prev) => ({
            ...prev,
            isLoading: false,
            isValidating: false,
            isError: true,
            error: apiError.message,
            errorDetails: apiError,
          }));

          onError?.(apiError);
        }
      }
    },
    [url, enabled, transform, onSuccess, onError, retryCount, retryDelay]
  );

  // Initial fetch
  useEffect(() => {
    mountedRef.current = true;
    fetchData();

    return () => {
      mountedRef.current = false;
      abortControllerRef.current?.abort();
    };
  }, [fetchData]);

  // Refetch interval
  useEffect(() => {
    if (refetchInterval <= 0 || !enabled) return;

    const intervalId = setInterval(() => {
      fetchData(true);
    }, refetchInterval);

    return () => clearInterval(intervalId);
  }, [fetchData, refetchInterval, enabled]);

  // Stale time tracking
  useEffect(() => {
    if (staleTime <= 0 || state.isStale || !state.lastUpdated) return;

    const timeoutId = setTimeout(() => {
      if (mountedRef.current) {
        setState((prev) => ({ ...prev, isStale: true }));
      }
    }, staleTime);

    return () => clearTimeout(timeoutId);
  }, [state.lastUpdated, staleTime, state.isStale]);

  // Actions
  const refetch = useCallback(async () => {
    retryCountRef.current = 0;
    await fetchData(true);
  }, [fetchData]);

  const mutate = useCallback((dataOrUpdater: T | ((prev: T | null) => T)) => {
    setState((prev) => ({
      ...prev,
      data: typeof dataOrUpdater === 'function'
        ? (dataOrUpdater as (prev: T | null) => T)(prev.data)
        : dataOrUpdater,
    }));
  }, []);

  const reset = useCallback(() => {
    setState({
      data: initialData,
      isLoading: false,
      isValidating: false,
      isError: false,
      error: null,
      errorDetails: null,
      isStale: true,
      lastUpdated: null,
    });
    retryCountRef.current = 0;
  }, [initialData]);

  const invalidate = useCallback(() => {
    if (cacheKey) {
      queryClient.invalidateQueries({ queryKey: [cacheKey] });
    }
    apiClient.invalidateCache(url ?? undefined);
    fetchData(true);
  }, [cacheKey, queryClient, url, fetchData]);

  return {
    ...state,
    refetch,
    mutate,
    reset,
    invalidate,
  };
}

// =============================================================================
// Streaming Hook
// =============================================================================

export interface UseStreamingDataOptions<T> {
  /** Transform each SSE message */
  transform?: (data: string) => T;
  /** Callback on each message */
  onMessage?: (data: T) => void;
  /** Callback on error */
  onError?: (error: Error) => void;
  /** Whether streaming is enabled */
  enabled?: boolean;
}

/**
 * Hook for streaming data via Server-Sent Events.
 *
 * @example
 * ```tsx
 * const { data, isConnected, reconnect } = useStreamingData<PredictionUpdate>(
 *   '/sports/predictions/stream',
 *   {
 *     transform: JSON.parse,
 *     onMessage: (update) => console.log('New prediction:', update),
 *   }
 * );
 * ```
 */
export function useStreamingData<T>(
  url: string | null,
  options: UseStreamingDataOptions<T> = {}
): {
  data: T[];
  isConnected: boolean;
  error: string | null;
  reconnect: () => void;
  clear: () => void;
} {
  const { transform, onMessage, onError, enabled = true } = options;

  const [data, setData] = useState<T[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cleanupRef = useRef<(() => void) | null>(null);

  const connect = useCallback(() => {
    if (!url || !enabled) return;

    // Cleanup existing connection
    cleanupRef.current?.();

    setIsConnected(false);
    setError(null);

    cleanupRef.current = apiClient.stream(
      url,
      (rawData) => {
        try {
          const parsed = transform ? transform(rawData) : (JSON.parse(rawData) as T);
          setData((prev) => [...prev, parsed]);
          setIsConnected(true);
          onMessage?.(parsed);
        } catch (err) {
          console.error('Stream parse error:', err);
        }
      }
    );

    setIsConnected(true);
  }, [url, enabled, transform, onMessage]);

  // Connect on mount
  useEffect(() => {
    connect();

    return () => {
      cleanupRef.current?.();
    };
  }, [connect]);

  const reconnect = useCallback(() => {
    setData([]);
    connect();
  }, [connect]);

  const clear = useCallback(() => {
    setData([]);
  }, []);

  return {
    data,
    isConnected,
    error,
    reconnect,
    clear,
  };
}

// =============================================================================
// Mutation Hook
// =============================================================================

export interface UseMutationOptions<TData, TVariables> {
  /** Callback on success */
  onSuccess?: (data: TData, variables: TVariables) => void;
  /** Callback on error */
  onError?: (error: APIError, variables: TVariables) => void;
  /** Callback on completion (success or error) */
  onSettled?: (data: TData | null, error: APIError | null, variables: TVariables) => void;
  /** Cache keys to invalidate on success */
  invalidateKeys?: string[];
}

export interface UseMutationResult<TData, TVariables> {
  mutate: (variables: TVariables) => void;
  mutateAsync: (variables: TVariables) => Promise<TData>;
  data: TData | null;
  isLoading: boolean;
  isError: boolean;
  error: string | null;
  reset: () => void;
}

/**
 * Hook for mutations (POST, PUT, DELETE).
 *
 * @example
 * ```tsx
 * const { mutate, isLoading } = useMutation<Order, OrderRequest>(
 *   'POST',
 *   '/orders',
 *   {
 *     onSuccess: (order) => toast.success(`Order ${order.id} placed!`),
 *     invalidateKeys: ['positions', 'portfolio'],
 *   }
 * );
 *
 * // Usage
 * mutate({ symbol: 'AAPL', quantity: 10, side: 'buy' });
 * ```
 */
export function useMutation<TData, TVariables = unknown>(
  method: 'POST' | 'PUT' | 'PATCH' | 'DELETE',
  url: string,
  options: UseMutationOptions<TData, TVariables> = {}
): UseMutationResult<TData, TVariables> {
  const { onSuccess, onError, onSettled, invalidateKeys = [] } = options;

  const [state, setState] = useState({
    data: null as TData | null,
    isLoading: false,
    isError: false,
    error: null as string | null,
  });

  const queryClient = useQueryClient();

  const mutateAsync = useCallback(
    async (variables: TVariables): Promise<TData> => {
      setState({ data: null, isLoading: true, isError: false, error: null });

      try {
        let response: TData;

        switch (method) {
          case 'POST':
            response = await apiClient.post<TData>(url, variables);
            break;
          case 'PUT':
            response = await apiClient.put<TData>(url, variables);
            break;
          case 'PATCH':
            response = await apiClient.patch<TData>(url, variables);
            break;
          case 'DELETE':
            response = await apiClient.delete<TData>(url);
            break;
        }

        setState({ data: response, isLoading: false, isError: false, error: null });

        // Invalidate caches
        invalidateKeys.forEach((key) => {
          queryClient.invalidateQueries({ queryKey: [key] });
          apiClient.invalidateCache(key);
        });

        onSuccess?.(response, variables);
        onSettled?.(response, null, variables);

        return response;
      } catch (err) {
        const apiError: APIError = isAPIError(err)
          ? err
          : {
              code: 500,
              name: 'MUTATION_ERROR',
              message: formatAPIError(err),
              timestamp: new Date().toISOString(),
            };

        setState({
          data: null,
          isLoading: false,
          isError: true,
          error: apiError.message,
        });

        onError?.(apiError, variables);
        onSettled?.(null, apiError, variables);

        throw apiError;
      }
    },
    [method, url, onSuccess, onError, onSettled, invalidateKeys, queryClient]
  );

  const mutate = useCallback(
    (variables: TVariables) => {
      mutateAsync(variables).catch(() => {
        // Error already handled in mutateAsync
      });
    },
    [mutateAsync]
  );

  const reset = useCallback(() => {
    setState({ data: null, isLoading: false, isError: false, error: null });
  }, []);

  return {
    mutate,
    mutateAsync,
    data: state.data,
    isLoading: state.isLoading,
    isError: state.isError,
    error: state.error,
    reset,
  };
}

// =============================================================================
// Exports
// =============================================================================

export default useAsyncData;
