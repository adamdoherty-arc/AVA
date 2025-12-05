/**
 * Enhanced Axios Instance
 * =======================
 *
 * Production-grade axios configuration with:
 * - Endpoint-aware timeout configuration
 * - Automatic retry with exponential backoff
 * - Circuit breaker pattern for failing endpoints
 * - Request cancellation support
 * - Enhanced error transformation
 * - Request deduplication for GET requests
 *
 * @author AVA Trading Platform
 * @updated 2025-12-05
 */

import axios from 'axios';
import type { AxiosInstance, AxiosRequestConfig, AxiosResponse, AxiosError, InternalAxiosRequestConfig } from 'axios';
import { API_BASE_URL } from '@/config/api';

// =============================================================================
// Timeout Configuration by Endpoint Type
// =============================================================================

/**
 * Endpoint timeout categories:
 * - FAST: Health checks, cached data (10s)
 * - STANDARD: Most API calls (30s)
 * - SLOW: External APIs like Robinhood, database queries (60s)
 * - AI: LLM/AI inference endpoints (120s)
 * - SCAN: Long-running scan operations (180s)
 */
const TIMEOUT_CONFIG = {
    FAST: 10000,      // 10 seconds
    STANDARD: 30000,  // 30 seconds
    SLOW: 60000,      // 60 seconds
    AI: 120000,       // 2 minutes
    SCAN: 180000,     // 3 minutes
} as const;

// Endpoint patterns for automatic timeout detection
const ENDPOINT_TIMEOUTS: Array<{ pattern: RegExp; timeout: number }> = [
    // Fast endpoints (cached, health checks)
    { pattern: /^\/health/, timeout: TIMEOUT_CONFIG.FAST },
    { pattern: /^\/cache\//, timeout: TIMEOUT_CONFIG.FAST },
    { pattern: /\/portfolio\/positions\/cached/, timeout: TIMEOUT_CONFIG.FAST }, // Database cache - instant
    { pattern: /\/portfolio\/sync\/status/, timeout: TIMEOUT_CONFIG.FAST }, // Just reading status

    // Slow endpoints (external APIs)
    { pattern: /\/portfolio\/positions(?!\/cached)/, timeout: TIMEOUT_CONFIG.SLOW }, // Live Robinhood
    { pattern: /\/portfolio\/sync\/trigger/, timeout: TIMEOUT_CONFIG.SLOW }, // Triggers Robinhood sync
    { pattern: /\/portfolio\/sync(?!\/status|\/trigger)/, timeout: TIMEOUT_CONFIG.SLOW },
    { pattern: /\/robinhood\//, timeout: TIMEOUT_CONFIG.SLOW },
    { pattern: /\/watchlist\//, timeout: TIMEOUT_CONFIG.SLOW },
    { pattern: /\/scanner\/watchlists/, timeout: TIMEOUT_CONFIG.SLOW },

    // AI endpoints
    { pattern: /\/ai\//, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/ai-picks/, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/agents\/.*\/invoke/, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/chat\//, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/deep-analysis/, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/portfolio-analysis/, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/recommendations/, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/consensus/, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/predictions\//, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/best-bets/, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/research\//, timeout: TIMEOUT_CONFIG.AI },
    { pattern: /\/sentiment/, timeout: TIMEOUT_CONFIG.AI },

    // Scan endpoints (longest operations)
    { pattern: /\/scanner\/scan/, timeout: TIMEOUT_CONFIG.SCAN },
    { pattern: /\/scanner\/stored-premiums/, timeout: TIMEOUT_CONFIG.SLOW },
    { pattern: /\/scanner\/dte(?:$|\?)/, timeout: TIMEOUT_CONFIG.SLOW }, // Match /scanner/dte or /scanner/dte?... but not /scanner/dte-comparison
    { pattern: /\/backtest/, timeout: TIMEOUT_CONFIG.SCAN },
    { pattern: /\/stress-test/, timeout: TIMEOUT_CONFIG.SCAN },
    { pattern: /\/var\?/, timeout: TIMEOUT_CONFIG.AI },
];

/**
 * Get appropriate timeout for an endpoint
 */
function getTimeoutForEndpoint(url: string): number {
    for (const { pattern, timeout } of ENDPOINT_TIMEOUTS) {
        if (pattern.test(url)) {
            return timeout;
        }
    }
    return TIMEOUT_CONFIG.STANDARD;
}

// =============================================================================
// Circuit Breaker
// =============================================================================

interface CircuitState {
    failures: number;
    lastFailure: number;
    isOpen: boolean;
}

class CircuitBreaker {
    private circuits = new Map<string, CircuitState>();
    private readonly threshold = 5;       // Failures before opening circuit
    private readonly resetTimeout = 30000; // Time before trying again (30s)

    private getCircuitKey(url: string): string {
        // Group by first path segment
        const match = url.match(/^\/([^/?]+)/);
        return match ? match[1] : 'default';
    }

    isOpen(url: string): boolean {
        const key = this.getCircuitKey(url);
        const state = this.circuits.get(key);

        if (!state || !state.isOpen) return false;

        // Check if reset timeout has passed
        if (Date.now() - state.lastFailure > this.resetTimeout) {
            state.isOpen = false;
            state.failures = 0;
            return false;
        }

        return true;
    }

    recordSuccess(url: string): void {
        const key = this.getCircuitKey(url);
        this.circuits.delete(key);
    }

    recordFailure(url: string): void {
        const key = this.getCircuitKey(url);
        const state = this.circuits.get(key) || { failures: 0, lastFailure: 0, isOpen: false };

        state.failures++;
        state.lastFailure = Date.now();

        if (state.failures >= this.threshold) {
            state.isOpen = true;
            console.warn(`[Circuit Breaker] Circuit opened for /${key} after ${state.failures} failures`);
        }

        this.circuits.set(key, state);
    }

    getStatus(): Record<string, { failures: number; isOpen: boolean }> {
        const status: Record<string, { failures: number; isOpen: boolean }> = {};
        for (const [key, state] of this.circuits) {
            status[key] = { failures: state.failures, isOpen: state.isOpen };
        }
        return status;
    }
}

const circuitBreaker = new CircuitBreaker();

// =============================================================================
// Request Deduplication
// =============================================================================

const pendingRequests = new Map<string, Promise<AxiosResponse>>();

function getRequestKey(config: AxiosRequestConfig): string {
    return `${config.method}:${config.url}:${JSON.stringify(config.params)}`;
}

// =============================================================================
// Retry Logic
// =============================================================================

interface RetryConfig {
    retries: number;
    retryDelay: number;
    retryCondition: (error: AxiosError) => boolean;
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
    retries: 2,
    retryDelay: 1000,
    retryCondition: (error: AxiosError) => {
        // Retry on network errors
        if (!error.response) return true;

        // Retry on 5xx errors and 429 (rate limit)
        const status = error.response.status;
        return status >= 500 || status === 429;
    },
};

async function withRetry<T>(
    fn: () => Promise<T>,
    config: RetryConfig = DEFAULT_RETRY_CONFIG
): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= config.retries; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error as Error;
            const axiosError = error as AxiosError;

            if (attempt < config.retries && config.retryCondition(axiosError)) {
                const delay = config.retryDelay * Math.pow(2, attempt);
                console.warn(`[Axios] Retry ${attempt + 1}/${config.retries} after ${delay}ms`);
                await new Promise(resolve => setTimeout(resolve, delay));
            } else {
                break;
            }
        }
    }

    throw lastError;
}

// =============================================================================
// Enhanced Error Types
// =============================================================================

export interface EnhancedError extends Error {
    isTimeout: boolean;
    isNetworkError: boolean;
    isCircuitOpen: boolean;
    statusCode?: number;
    endpoint?: string;
    retryable: boolean;
}

function createEnhancedError(error: AxiosError, isCircuitOpen = false): EnhancedError {
    const enhanced = new Error(error.message) as EnhancedError;
    enhanced.name = 'APIError';
    enhanced.isTimeout = error.code === 'ECONNABORTED' || error.message.includes('timeout');
    enhanced.isNetworkError = !error.response && !enhanced.isTimeout;
    enhanced.isCircuitOpen = isCircuitOpen;
    enhanced.statusCode = error.response?.status;
    enhanced.endpoint = error.config?.url;
    enhanced.retryable = !error.response || error.response.status >= 500 || error.response.status === 429;

    // Preserve original error properties
    Object.assign(enhanced, {
        response: error.response,
        config: error.config,
        code: error.code,
    });

    return enhanced;
}

// =============================================================================
// Axios Instance
// =============================================================================

export const axiosInstance: AxiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: TIMEOUT_CONFIG.STANDARD,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor
axiosInstance.interceptors.request.use(
    (config: InternalAxiosRequestConfig) => {
        const url = config.url || '';

        // Check circuit breaker
        if (circuitBreaker.isOpen(url)) {
            const error = new Error(`Circuit breaker open for ${url}`) as EnhancedError;
            error.isCircuitOpen = true;
            error.isTimeout = false;
            error.isNetworkError = false;
            error.retryable = false;
            error.endpoint = url;
            // Log circuit breaker rejection with proper context
            console.warn(`🔌 [Circuit Breaker] Blocked ${config.method?.toUpperCase() || 'GET'} ${url}`);
            return Promise.reject(error);
        }

        // Auto-detect timeout based on endpoint if not explicitly set
        if (!config.timeout || config.timeout === TIMEOUT_CONFIG.STANDARD) {
            config.timeout = getTimeoutForEndpoint(url);
        }

        // Add correlation ID for request tracing
        config.headers['X-Correlation-ID'] = crypto.randomUUID();

        // Store request start time for logging
        (config as AxiosRequestConfig & { metadata?: { startTime: number } }).metadata = {
            startTime: Date.now(),
        };

        return config;
    },
    (error) => Promise.reject(error)
);

// Response interceptor
axiosInstance.interceptors.response.use(
    (response: AxiosResponse) => {
        // Record success for circuit breaker
        const url = response.config.url || '';
        circuitBreaker.recordSuccess(url);

        // Log request duration in development
        const metadata = (response.config as AxiosRequestConfig & { metadata?: { startTime: number } }).metadata;
        if (metadata?.startTime) {
            const duration = Date.now() - metadata.startTime;
            const emoji = duration > 5000 ? '🐢' : duration > 1000 ? '⏱️' : '⚡';
            console.debug(`${emoji} [API] ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status} (${duration}ms)`);
        }

        return response;
    },
    async (error: AxiosError) => {
        // Check if this is already a circuit breaker error (rejected in request interceptor)
        if ((error as EnhancedError).isCircuitOpen) {
            return Promise.reject(error);
        }

        const url = error.config?.url || '';

        // Record failure for circuit breaker
        circuitBreaker.recordFailure(url);

        // Enhanced error logging
        const isTimeout = error.code === 'ECONNABORTED' || error.message.includes('timeout');
        const message = error.response?.data
            ? (error.response.data as { error?: { message?: string }; detail?: string }).error?.message
                || (error.response.data as { detail?: string }).detail
            : error.message;

        const emoji = isTimeout ? '⏰' : error.response ? '❌' : '🔌';
        console.error(`${emoji} [API Error] ${error.config?.method?.toUpperCase() || 'GET'} ${url}: ${message}`);

        // Transform to enhanced error
        const enhancedError = createEnhancedError(error);
        return Promise.reject(enhancedError);
    }
);

// =============================================================================
// Enhanced Request Methods with Deduplication
// =============================================================================

/**
 * Make a GET request with automatic deduplication
 * Identical concurrent GET requests will share the same response
 */
export async function deduplicatedGet<T>(
    url: string,
    config?: AxiosRequestConfig
): Promise<AxiosResponse<T>> {
    const requestConfig = { ...config, url, method: 'GET' as const };
    const key = getRequestKey(requestConfig);

    // Check for pending identical request
    const pending = pendingRequests.get(key);
    if (pending) {
        console.debug(`[Axios] Deduplicating request to ${url}`);
        return pending as Promise<AxiosResponse<T>>;
    }

    // Make the request
    const promise = axiosInstance.get<T>(url, config).finally(() => {
        pendingRequests.delete(key);
    });

    pendingRequests.set(key, promise as Promise<AxiosResponse>);
    return promise;
}

/**
 * Make a request with automatic retry
 */
export async function retryableRequest<T>(
    config: AxiosRequestConfig,
    retryConfig?: Partial<RetryConfig>
): Promise<AxiosResponse<T>> {
    return withRetry(
        () => axiosInstance.request<T>(config),
        { ...DEFAULT_RETRY_CONFIG, ...retryConfig }
    );
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Get circuit breaker status
 */
export function getCircuitBreakerStatus() {
    return circuitBreaker.getStatus();
}

/**
 * Check if an error is a timeout error
 */
export function isTimeoutError(error: unknown): boolean {
    if (error instanceof Error) {
        return (error as EnhancedError).isTimeout === true
            || error.message.includes('timeout')
            || (error as AxiosError).code === 'ECONNABORTED';
    }
    return false;
}

/**
 * Check if an error is retryable
 */
export function isRetryableError(error: unknown): boolean {
    if (error instanceof Error) {
        return (error as EnhancedError).retryable === true;
    }
    return false;
}

/**
 * Format error message for display
 */
export function formatErrorMessage(error: unknown): string {
    if (error instanceof Error) {
        const enhanced = error as EnhancedError;
        if (enhanced.isTimeout) {
            return 'Request timed out. The server may be busy.';
        }
        if (enhanced.isNetworkError) {
            return 'Network error. Please check your connection.';
        }
        if (enhanced.isCircuitOpen) {
            return 'Service temporarily unavailable. Please try again later.';
        }
        return enhanced.message;
    }
    return 'An unexpected error occurred';
}

// =============================================================================
// Exports
// =============================================================================

export { TIMEOUT_CONFIG };
export type { EnhancedError };

// Re-export the modern API client for progressive migration
export { apiClient, APIClient, isAPIError, formatAPIError } from './api-client';
export type { APIError, APIResponse, APIErrorResponse, RequestOptions, APIClientConfig } from './api-client';
